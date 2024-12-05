from datasets import load_dataset
from diffusers import AutoencoderKL
from util import preprocess_image
import torch
import os
from tqdm import tqdm

device = "cuda:0"

dataset = load_dataset("huggan/smithsonian_butterflies_subset")

vae = AutoencoderKL.from_single_file("F:/models/vae/vae-ft-mse-840000-ema-pruned.ckpt").to(device)

# Prepare directory to save the resized dataset
save_dir = "datasets/resized_butterflies_vae_dataset"
os.makedirs(save_dir, exist_ok=True)


def save_resized_dataset(dataset):
    with torch.no_grad():
        for split in dataset:
            split_dir = os.path.join(save_dir, split)
            os.makedirs(split_dir, exist_ok=True)

            # Process each example in the dataset
            for i, example in enumerate(tqdm(dataset[split])):
                image = example["image"]
                image = preprocess_image(image.convert("RGB"))

                # Encode using VAE
                encoded = vae.encode(image.unsqueeze(0).to(device), return_dict=False)[0].sample().squeeze()
                encoded = encoded.cpu().numpy()

                # Save tensor as a .pt file
                tensor_file = os.path.join(split_dir, f"{i}.pt")
                torch.save(encoded, tensor_file)

                torch.cuda.empty_cache()  # Clear unused memory


# Save the resized dataset
save_resized_dataset(dataset)
