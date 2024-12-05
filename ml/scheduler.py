import torch
import numpy as np

from util import match_shape, device


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
