import argparse
import cv2
import torch
import numpy as np
import moviepy.editor as mpy
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from torchvision.transforms import ToTensor
from tqdm import tqdm
from PIL import Image


def nearest_multiple_of_eight(x):
    """Round up to the nearest multiple of 8."""
    return int(np.ceil(x / 8) * 8)


def resize_to_minimum_512(image):
    """
    Resize image to have at least one side 512 pixels while maintaining aspect ratio.
    Then ensure dimensions are multiples of 8.

    Returns:
    - Resized image
    - Original image size for later downscaling
    """
    height, width = image.shape[:2]
    aspect_ratio = width / height

    # Determine scale factor to make the smaller side 512
    if width < height:
        new_width = 512
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 512
        new_width = int(new_height * aspect_ratio)

    # Resize to 512
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # Ensure multiple of 8
    new_width = nearest_multiple_of_eight(new_width)
    new_height = nearest_multiple_of_eight(new_height)

    # Final resize to multiple of 8
    final_resized = cv2.resize(resized_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return final_resized, (height, width)


def process_video(input_video, output_video, sd_checkpoint, prompt, imagination, device):
    # Load Stable Diffusion model
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(sd_checkpoint).to(device, torch_dtype=torch.float16)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Open the input video
    video_capture = cv2.VideoCapture(input_video)
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Prepare to store processed frames
    processed_frames = []

    # Process each frame
    progress_bar = tqdm(total=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame to at least 512 pixels
        frame_resized, original_size = resize_to_minimum_512(frame)

        # Convert color space
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Calculate prompt strength based on imagination
        prompt_strength = imagination

        # Run img2img
        with torch.no_grad():
            result = pipe(
                prompt=prompt, image=pil_image, strength=prompt_strength, guidance_scale=7.5, num_inference_steps=15
            ).images[0]

        # Convert result back to numpy array
        result_array = np.array(result)

        # Resize back to original size
        result_resized = cv2.resize(
            result_array, (original_size[1], original_size[0]), interpolation=cv2.INTER_LANCZOS4
        )

        # Convert back to BGR for video writing
        result_bgr = cv2.cvtColor(result_resized, cv2.COLOR_RGB2BGR)

        processed_frames.append(result_bgr)
        progress_bar.update(1)

    progress_bar.close()
    video_capture.release()

    # Create and write video using moviepy
    clip = mpy.ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames], fps=fps)
    clip.write_videofile(output_video, codec="libx264")

    print(f"Processed video saved to {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Process a video with Stable Diffusion img2img.")
    parser.add_argument("--input-video", type=str, required=True, help="Path to the input MP4 video.")
    parser.add_argument("--sd-checkpoint", type=str, required=True, help="Path to the Stable Diffusion checkpoint.")
    parser.add_argument("--output-video", type=str, required=True, help="Path to the output MP4 video.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for img2img transformation.")
    parser.add_argument("--imagination", type=float, default=0.0, help="Prompt strength (0 to 1).")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_video(
        input_video=args.input_video,
        output_video=args.output_video,
        sd_checkpoint=args.sd_checkpoint,
        prompt=args.prompt,
        imagination=args.imagination,
        device=device,
    )


if __name__ == "__main__":
    main()
