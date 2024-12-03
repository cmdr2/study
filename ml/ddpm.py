import os
import torch
from diffusers import DDPMScheduler as DiffusersDDPMScheduler
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import safetensors.torch
from datasets import load_dataset
import numpy as np

from util import (
    save_image_grid,
    get_image_tensor,
    match_shape,
    preprocess_image,
    denormalize_image,
    PREPROCESSED_IMAGE_SIZE,
)

device = "cuda:0"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
LR_WARMUP_STEPS = 500
TRAIN_BATCH_SIZE = 16


# I don't understand the math in the scheduler, but that's what's used by diffusers and elsewhere.
# It basically adds noise (after scaling) or removes noise, based on the timestep and beta config.
class DDPMScheduler:
    def __init__(self, num_train_timesteps=10, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        betas = torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps).to(device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        self.alphas_cumprod_sqrt = self.alphas_cumprod**0.5
        self.alphas_cumprod_one_minus_sqrt = (1.0 - self.alphas_cumprod) ** 0.5
        self.alphas_cumprod_one_by_sqrt = (1.0 + self.alphas_cumprod) ** (-0.5)
        self.beta_by_alpha_cumprod = betas / self.alphas_cumprod_one_minus_sqrt

        self.timesteps = None
        self.num_inference_steps = None

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor):
        # from the video: https://www.youtube.com/watch?v=w8YQcEd77_o
        alphas_cumprod_sqrt = self.alphas_cumprod_sqrt[timesteps]
        alphas_cumprod_one_minus_sqrt = self.alphas_cumprod_one_minus_sqrt[timesteps]

        alphas_cumprod_sqrt = match_shape(alphas_cumprod_sqrt, original_samples)
        alphas_cumprod_one_minus_sqrt = match_shape(alphas_cumprod_one_minus_sqrt, original_samples)

        return alphas_cumprod_sqrt * original_samples + alphas_cumprod_one_minus_sqrt * noise

    def step(self, noise_pred: torch.Tensor, timesteps: torch.IntTensor, noisy_samples: torch.Tensor, **kwargs):
        # this math doesn't work (from the video: https://www.youtube.com/watch?v=w8YQcEd77_o):
        # alphas_cumprod_one_by_sqrt = self.alphas_cumprod_one_by_sqrt[timesteps]
        # beta_by_alpha_cumprod = self.beta_by_alpha_cumprod[timesteps]

        # x = alphas_cumprod_one_by_sqrt * (noisy_samples - beta_by_alpha_cumprod * noise_pred)  # + something
        # return x, None

        # this math works (from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L399)
        t = timesteps
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor([1.0]).to(device)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        curr_alpha_t = alpha_prod_t / alpha_prod_t_prev
        curr_beta_t = 1 - curr_alpha_t

        pred_orig_sample = (noisy_samples - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        pred_orig_sample = pred_orig_sample.clamp(-1.0, 1.0)

        pred_orig_sample_coeff = (alpha_prod_t_prev**0.5) * curr_beta_t / beta_prod_t
        curr_sample_coeff = (curr_alpha_t**0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_orig_sample_coeff * pred_orig_sample + curr_sample_coeff * noisy_samples

        variance_noise = torch.randn(noise_pred.shape).to(device)
        variance = (self._get_variance(t) ** 0.5) * variance_noise

        pred_prev_sample += variance

        return pred_prev_sample, None

    def set_timesteps(self, num_inference_steps):
        step = self.num_train_timesteps // num_inference_steps
        self.timesteps = (np.arange(0, num_inference_steps) * step).round()[::-1].astype(np.int64)
        self.timesteps = torch.from_numpy(self.timesteps)

        self.num_inference_steps = num_inference_steps

    def previous_timestep(self, timestep):
        return timestep - self.num_train_timesteps // self.num_inference_steps

    def _get_variance(self, t):
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor([1.0]).to(device)
        curr_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * curr_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance


# load dataset
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")


def transform(images):
    images = [preprocess_image(image.convert("RGB")) for image in images["image"]]
    return {"images": images}


dataset.set_transform(transform)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)


sample_image = dataset[0]["images"].unsqueeze(0)
channels, size = sample_image.shape[1], sample_image.shape[2]

print("sample image", sample_image.shape)

model = UNet2DModel(
    sample_size=128,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

model = model.to(device)

# scheduler = DiffusersDDPMScheduler(num_train_timesteps=1000)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# print(images.shape)
# x = model(images, timestep=0).sample
# print(x.shape)


# load the current checkpoint
def load_model():
    if os.path.exists("model.sft"):
        sd = safetensors.torch.load_file("model.sft")
        model.load_state_dict(sd)
        del sd


# train
def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=LR_WARMUP_STEPS, num_training_steps=len(train_dataloader) * NUM_EPOCHS
    )

    global_step = 0

    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch: {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(device)
            batch_size = clean_images.shape[0]

            noise = torch.randn(clean_images.shape).to(device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,)).to(device)

            noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps).sample

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1


# save the model
def save_model():
    safetensors.torch.save_file(model.state_dict(), "model.sft")


# evaluate
def evaluate():
    img = torch.randn(sample_image.shape)
    img = img.to(device)

    num_inference_steps = 25

    samples = []

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        noise_pred = model(img, timestep=t).sample
        img = scheduler.step(noise_pred, torch.IntTensor([t]), img, return_dict=False)[0]

        img_out = (img / 2 + 0.5).clamp(0, 1)
        samples.append(img_out.squeeze())

    save_image_grid([samples], "out_grid.jpg")

    img = img.squeeze()

    img_pil = denormalize_image(img)
    img_pil.save("out.jpg")


if __name__ == "__main__":
    load_model()
    train()
    save_model()
    evaluate()