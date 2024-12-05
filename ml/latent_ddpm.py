# Run preprocess_butterfly_dataset.py first

import os
import torch
from diffusers import DDPMScheduler as DiffusersDDPMScheduler, AutoencoderKL
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm
import safetensors.torch
from torch.utils.data import DataLoader, Dataset
from glob import glob

from util import save_image_grid, denormalize_image, device
from scheduler import DDPMScheduler

MODEL_FILE = "trained_models/butterfly_model_latent.sft"


LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
LR_WARMUP_STEPS = 500
TRAIN_BATCH_SIZE = 16
num_train_timesteps = 1000


class FileDatasetLoader(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception("Can't find dataset dir!")
        self.image_files = glob(os.path.join(data_dir, "*.pt"))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = torch.load(image_file)
        return image


vae = AutoencoderKL.from_single_file("F:/models/vae/vae-ft-mse-840000-ema-pruned.ckpt").to(device)

# load dataset
dataset = FileDatasetLoader("datasets/resized_butterflies_vae_dataset/train")

print("Dataset size:", len(dataset))

train_dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

sample_image = torch.tensor(dataset[0])
channels, size = sample_image.shape[0], sample_image.shape[1]

print("sample image", sample_image.shape)
save_image_grid([sample_image.unsqueeze(0)], "out/bar0.jpg")

model = UNet2DModel(
    sample_size=size,
    in_channels=channels,
    out_channels=channels,
    layers_per_block=2,
    norm_num_groups=size // 2,
    block_out_channels=(size, size, size * 2, size * 2, size * 4, size * 4),
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

# scheduler = DiffusersDDPMScheduler(num_train_timesteps=num_train_timesteps)
scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

# print(sample_image.shape)
# sample_image = sample_image.to(device)
# x = model(sample_image, timestep=0).sample
# print(x.shape)


# load the current checkpoint
def load_model():
    if os.path.exists(MODEL_FILE):
        sd = safetensors.torch.load_file(MODEL_FILE)
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

        for batch in train_dataloader:
            clean_images = batch.to(device)
            batch_size = clean_images.shape[0]

            train_image = clean_images

            noise = torch.randn(train_image.shape).to(device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,)).to(device)

            noisy_images = scheduler.add_noise(train_image, noise, timesteps)
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

            batch.to("cpu")


# save the model
def save_model():
    safetensors.torch.save_file(model.state_dict(), MODEL_FILE)


# evaluate
def evaluate():
    with torch.no_grad():
        img = torch.randn((1, 4, size, size))
        img = img.to(device)

        num_inference_steps = 25

        samples = []

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            noise_pred = model(img, timestep=t).sample
            img = scheduler.step(noise_pred, torch.IntTensor([t]), img, return_dict=False)[0]

            img_out = (img / 2 + 0.5).clamp(0, 1)
            # img_out = vae.decode(img_out).sample
            samples.append(img_out.squeeze())

        img = vae.decode(img).sample.squeeze()
        print(img.shape)

        save_image_grid([samples], "out/out_grid.jpg")

        img = img.squeeze()

        img_pil = denormalize_image(img)
        img_pil.save("out/out.jpg")


if __name__ == "__main__":
    load_model()
    train()
    save_model()
    evaluate()
