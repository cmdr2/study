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
    device,
)
from scheduler import DDPMScheduler


MODEL_FILE = "trained_models/butterfly_model_non_latent.sft"

LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
LR_WARMUP_STEPS = 500
TRAIN_BATCH_SIZE = 16


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
    safetensors.torch.save_file(model.state_dict(), MODEL_FILE)


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

    save_image_grid([samples], "out/out_grid.jpg")

    img = img.squeeze()

    img_pil = denormalize_image(img)
    img_pil.save("out/out.jpg")


if __name__ == "__main__":
    load_model()
    train()
    save_model()
    evaluate()
