import torch
import math
import torch.nn as nn
import numpy as np


class ConsistencyModelSampler:
    def __init__(self, noise_scheduler, scale_ranges):
        """
        Initialize consistency model sampler with the original noise scheduler

        Args:
            noise_scheduler: Original linear noise scheduler from LDM
            scale_ranges: Subset of timesteps used in consistency model training
        """
        self.noise_scheduler = noise_scheduler

        # If no scale ranges provided, use all timesteps
        if scale_ranges is None:
            self.scale_ranges = list(range(0, noise_scheduler.num_timesteps, 20))
        else:
            self.scale_ranges = sorted(scale_ranges)

        # Precompute alpha and noise values for selected timesteps
        self.selected_alphas = self.noise_scheduler.alpha_cum_prod[self.scale_ranges]
        self.selected_sqrt_alphas = torch.sqrt(self.selected_alphas)
        self.selected_sqrt_one_minus_alphas = torch.sqrt(1 - self.selected_alphas)

    def sample(
            self,
            consistency_model: nn.Module,
            vae: nn.Module,
            latent_shape: tuple[int, ...],
            cond_input: dict = None,
            num_steps: int = None,
            device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Sample images using the consistency model while respecting the original noise schedule

        Args:
            consistency_model: Trained consistency model
            vae: VAE decoder
            latent_shape: Shape of initial latent noise
            cond_input: Conditioning input
            num_steps: Number of sampling steps (defaults to scale ranges length)
            device: Device to run sampling on

        Returns:
            Generated images
        """
        # Use scale ranges length if no specific num_steps provided
        if num_steps is None:
            num_steps = len(self.scale_ranges)

        # Select subset of scale ranges for sampling
        sample_timesteps = self.scale_ranges[-num_steps:]

        # Initialize noise
        noise = torch.randn(latent_shape, device=device)
        noise *= self.selected_sqrt_alphas[-1]

        # Iterative sampling following noise schedule
        for i in range(len(sample_timesteps) - 1, 0, -1):
            t_cur = sample_timesteps[i]

            # Current noise schedule parameters
            alpha_cur = self.selected_alphas[i]
            sqrt_alpha_cur = self.selected_sqrt_alphas[i]
            sqrt_one_minus_alpha_cur = self.selected_sqrt_one_minus_alphas[i]

            # Previous noise schedule parameters
            alpha_prev = self.selected_alphas[i - 1]
            sqrt_alpha_prev = self.selected_sqrt_alphas[i - 1]
            sqrt_one_minus_alpha_prev = self.selected_sqrt_one_minus_alphas[i - 1]

            # Create timestep tensor
            t_tensor = torch.ones((latent_shape[0],), device=device, dtype=torch.long) * t_cur

            # Get model prediction
            noise_pred = consistency_model(noise, t_tensor, cond_input)

            # Predict x0 (original image) using current noise prediction
            x0_pred = (noise - sqrt_one_minus_alpha_cur * noise_pred) / sqrt_alpha_cur
            x0_pred = torch.clamp(x0_pred, -1., 1.)

            # Compute coefficients for interpolation
            c1 = sqrt_alpha_prev / sqrt_alpha_cur
            c2 = sqrt_one_minus_alpha_prev - c1 * sqrt_one_minus_alpha_cur

            # Update noise
            noise = c1 * noise + c2 * x0_pred

            # Add noise for next step if not final step
            if i > 1:
                noise_scale = math.sqrt(alpha_prev - alpha_cur)
                noise += noise_scale * torch.randn_like(noise)

        # Final denoising step
        t_tensor = torch.ones((latent_shape[0],), device=device, dtype=torch.long) * sample_timesteps[0]
        noise_pred = consistency_model(noise, t_tensor, cond_input)
        noise = (noise - self.selected_sqrt_one_minus_alphas[0] * noise_pred) / self.selected_sqrt_alphas[0]

        # Decode final latent to image
        generated_images = vae.decode(noise)
        return generated_images


class LossAwareSampler:
    def __init__(self, timesteps):
        """
        Initialize the Loss-Aware Sampler with the given timesteps.
        """
        self.timesteps = timesteps
        self.losses = np.zeros(timesteps, dtype=np.float32)  # Initialize losses

    def update_with_all_losses(self, ts, losses):
        """
        Update the sampler with new timesteps and losses.
        """
        for t, loss in zip(ts, losses):
            idx = np.where(self.timesteps == t.detach().item())[0]
            if idx.size > 0:
                self.losses[idx[0]] = loss

    def sample(self, batch_size):
        """
        Sample timesteps based on the updated losses.
        """
        if self.timesteps == 0:
            raise ValueError("Timesteps array is empty!")

        # Ensure losses are positive
        adjusted_losses = np.maximum(self.losses, 1e-6)  # Avoid division by zero
        probabilities = adjusted_losses / np.sum(adjusted_losses)

        # Log a warning if probabilities are invalid
        if np.isnan(probabilities).any() or len(probabilities) == 0:
            raise ValueError("Invalid probabilities! Check your loss values and update logic.")

        return np.random.choice(self.timesteps, size=batch_size, p=probabilities)
