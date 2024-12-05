import os
import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob

device = "cuda:0"


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


dataset = FileDatasetLoader("datasets/resized_butterflies_vae_dataset/train")

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

sample_image = dataset[0]
channels, size = sample_image.shape[0], sample_image.shape[1]

# size = size // 8

# save_image_grid(sample_image, "out/bar0.jpg")
print("sample image", sample_image.shape)

# Global variable to hold the sum of all tensor values
global_sum = 0.0


# Function to sum tensor values
def sum_tensor_values(dataloader):
    global global_sum
    for batch in dataloader:
        # Sum values of the batch and add to the global sum
        print(batch.shape, type(batch))
        global_sum += batch.sum().item()


# Run the sum function on both datasets
sum_tensor_values(train_loader)

# Print the result
print("Total sum of tensor values:", global_sum)
