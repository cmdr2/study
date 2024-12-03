from PIL import Image
from torchvision import transforms

ORIG_SIZE = 512
PREPROCESSED_IMAGE_SIZE = 128

pil_to_tensor_transform = transforms.ToTensor()
preprocess_image = transforms.Compose(
    [
        transforms.Resize((PREPROCESSED_IMAGE_SIZE, PREPROCESSED_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
denormalize_image = transforms.Compose(
    [
        transforms.Normalize([-1], [2]),  # Reverse normalization: (x - mean) / std => x * std + mean
        transforms.ToPILImage(),
        transforms.Resize((ORIG_SIZE, ORIG_SIZE)),
    ]
)


def save_image_grid(image_tensor_grid, output_path):
    """
    Saves a 2D grid of image tensors as a single image file.

    Args:
    - image_tensor_grid (list of list of torch.Tensor): 2D array of image tensors.
    - output_path (str): Path to save the final image file.

    Returns:
    - None
    """
    # First, convert the 2D grid of tensors to a list of PIL images
    pil_images = []
    for row in image_tensor_grid:
        pil_row = []
        for tensor in row:
            # Convert each tensor to a PIL image
            pil_image = transforms.ToPILImage()(tensor.clamp(0, 1))  # Clamp values to [0, 1] range
            pil_row.append(pil_image)
        pil_images.append(pil_row)

    # Determine the total width and height of the final image
    total_width = sum([img.width for img in pil_images[0]])  # Sum of the widths of all images in the first row
    total_height = sum([row[0].height for row in pil_images])  # Sum of the heights of all images in the first column

    # Create a new blank image to combine all the images
    combined_image = Image.new("RGB", (total_width, total_height))

    # Paste each image in the grid into the combined image
    y_offset = 0
    for row in pil_images:
        x_offset = 0
        for img in row:
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += row[0].height  # All images in the row should have the same height

    # Save the final image
    combined_image.save(output_path)
    print(f"Image saved to {output_path}")


def get_image_tensor(path):
    image = Image.open(path)
    image_tensor = pil_to_tensor_transform(image)
    return image_tensor


def match_shape(a, b):
    while len(a.shape) < len(b.shape):
        a = a.unsqueeze(-1)

    return a
