import argparse
import cv2
import torch
import numpy as np
import moviepy.editor as mpy
from diffusers import AutoencoderKL
from torchvision.transforms import ToTensor
from tqdm import tqdm


def process_video(input_path, output_path, vae_checkpoint, imagination, device):
    # Load VAE model
    vae = AutoencoderKL.from_single_file(vae_checkpoint).to(device).eval()

    # Open the input video
    video_capture = cv2.VideoCapture(input_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare to store processed frames
    processed_frames = []

    # Process each frame
    to_tensor = ToTensor()
    progress_bar = tqdm(total=int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing video")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert frame to tensor and normalize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = to_tensor(frame_rgb).unsqueeze(0).to(device)

        # Encode and decode with VAE
        with torch.no_grad():
            latent = vae.encode(frame_tensor * 2 - 1).latent_dist.sample()

            # Add noise scaled by imagination to the latent representation
            noise = torch.randn_like(latent) * imagination
            noisy_latent = latent + noise

            # Decode the latent representation
            reconstructed = vae.decode(noisy_latent).sample
            reconstructed = (reconstructed.clamp(-1, 1) + 1) / 2

        # Convert back to image
        reconstructed_image = (reconstructed.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        reconstructed_bgr = cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)

        processed_frames.append(reconstructed_bgr)
        progress_bar.update(1)

    progress_bar.close()
    video_capture.release()

    # Create and write video using moviepy
    clip = mpy.ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in processed_frames], fps=fps)
    clip.write_videofile(output_path, codec="libx264")

    print(f"Processed video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video with a VAE encode-decode step.")
    parser.add_argument("input_video", type=str, help="Path to the input MP4 video.")
    parser.add_argument("vae_checkpoint", type=str, help="Path to the VAE checkpoint.")
    parser.add_argument("output_video", type=str, help="Path to the output MP4 video.")
    parser.add_argument("--imagination", type=float, default=0.0, help="Imagination level (0 to 1).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_video(args.input_video, args.output_video, args.vae_checkpoint, args.imagination, device)
