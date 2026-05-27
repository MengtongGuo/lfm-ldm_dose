import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from denoiser.karras_denoiser import KarrasDiffusion
from models.pl_ldm_2 import LightningLatentDiffusion
from utils.diffusion_utils import append_dims


class LightningConsistencyModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,  # Student model
        diffusion: KarrasDiffusion,
        vae_model: nn.Module,
        teacher_model: LightningLatentDiffusion = None,
        teacher_diffusion: KarrasDiffusion = None,
        training_mode="consistency_distillation",  # Options: "consistency_distillation", "consistency_training", "progdist"
        loss_norm="l2",
        weight_schedule="uniform",
        lr=1e-4,
        weight_decay=0.0,
    ):
        super().__init__()
        self.model = model
        self.vae = vae_model
        self.diffusion = diffusion
        self.teacher_model = teacher_model
        self.teacher_diffusion = teacher_diffusion or diffusion
        self.training_mode = training_mode
        self.loss_norm = loss_norm
        self.weight_schedule = weight_schedule
        self.lr = lr
        self.weight_decay = weight_decay

        if teacher_model:
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()

    def forward(self, noisy_latents, t, cond_input=None):
        return self.model(noisy_latents, t, cond_input)

    def training_step(self, batch, batch_idx):
        images, cond_input = batch
        images = images.float()

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        # Compute loss
        if self.training_mode == "consistency_training":
            loss = self._consistency_losses(z, cond_input)
        elif self.training_mode == "consistency_distillation":
            loss = self._consistency_losses(z, cond_input, distillation=True)
        elif self.training_mode == "progdist":
            loss = self._progdist_losses(z, cond_input)
        else:
            raise ValueError(f"Unsupported training mode: {self.training_mode}")

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def _consistency_losses(self, z, cond_input, distillation=False):
        """
        Compute consistency losses, optionally using distillation from a teacher model.
        """
        noise = torch.randn_like(z)
        num_scales = self.diffusion.steps
        dims = z.ndim

        # Sample random timestep indices
        indices = torch.randint(0, num_scales - 1, (z.shape[0],), device=self.device)

        t = self._compute_noise_schedule(indices)
        t2 = self._compute_noise_schedule(indices + 1)

        # Add noise
        x_t = z + noise * append_dims(t, dims)

        # Denoise functions
        student_denoised = self.diffusion.get_denoised_latent(self.model, x_t, t, cond_input)

        if distillation:
            # Use teacher to compute the next noisy step
            x_t2 = self._heun_solver(x_t, t, t2, z)
            with torch.no_grad():
                teacher_denoised = self.teacher_diffusion.get_denoised_latent(
                    self.teacher_model, x_t2, t2, cond_input
                )
            target = teacher_denoised.detach()
        else:
            # Perform Euler solver for next step without teacher
            x_t2 = self._euler_solver(x_t, t, t2, z)
            target = x_t2

        # Compute loss
        loss = self._compute_loss(student_denoised, target, t)
        return loss

    def _progdist_losses(self, z, cond_input):
        """
        Compute progressive distillation losses for the student model.
        """
        noise = torch.randn_like(z)
        num_scales = self.diffusion.steps
        dims = z.ndim

        # Sample random timestep indices
        indices = torch.randint(0, num_scales - 1, (z.shape[0],), device=self.device)

        t = self._compute_noise_schedule(indices)
        t2 = self._compute_noise_schedule(indices + 0.5)
        t3 = self._compute_noise_schedule(indices + 1)

        # Add noise
        x_t = z + noise * append_dims(t, dims)

        # Student's prediction
        student_denoised = self.diffusion.get_denoised_latent(self.model, x_t, t, cond_input)

        # Teacher's prediction for t2 and t3
        x_t2 = self._euler_solver(x_t, t, t2)
        x_t3 = self._euler_solver(x_t2, t2, t3)
        target = self._euler_to_denoiser(x_t, t, x_t3, t3)

        # Compute loss
        loss = self._compute_loss(student_denoised, target, t)
        return loss

    def _euler_solver(self, samples, t, next_t, x0=None):
        """
        Euler solver to compute the next noisy sample.
        """
        denoiser = x0 if x0 is not None else samples
        d = (samples - denoiser) / append_dims(t, samples.ndim)
        return samples + d * append_dims(next_t - t, samples.ndim)

    def _heun_solver(self, samples, t, next_t, x0=None):
        """
        Heun solver to compute the next noisy sample.
        """
        x = samples
        denoiser = x0 if x0 is not None else x
        d = (x - denoiser) / append_dims(t, x.ndim)
        x = x + d * append_dims(next_t - t, x.ndim)
        next_d = (x - denoiser) / append_dims(next_t, x.ndim)
        return x + (d + next_d) * append_dims((next_t - t) / 2, x.ndim)

    def _euler_to_denoiser(self, x_t, t, x_next_t, next_t):
        """
        Converts Euler step outputs to a denoised image.
        """
        return x_t - append_dims(t, x_t.ndim) * (x_next_t - x_t) / append_dims(next_t - t, x_t.ndim)

    def _compute_loss(self, student, target, t):
        """
        Compute weighted loss based on SNR or uniform weighting.
        """
        snrs = self.diffusion.get_snr(t)
        weights = self._get_weightings(snrs)

        if self.loss_norm == "l1":
            diffs = torch.abs(student - target)
            loss = diffs.mean() * weights
        elif self.loss_norm == "l2":
            diffs = (student - target) ** 2
            loss = diffs.mean() * weights
        else:
            raise ValueError(f"Unsupported loss norm: {self.loss_norm}")
        return loss

    def _get_weightings(self, snrs):
        """
        Compute weights based on the specified schedule.
        """
        if self.weight_schedule == "uniform":
            return torch.ones_like(snrs)
        elif self.weight_schedule == "snr":
            return snrs
        else:
            raise NotImplementedError(f"Unknown weight schedule: {self.weight_schedule}")

    def _compute_noise_schedule(self, indices):
        """
        Compute noise levels based on the noise schedule.
        """
        rho = self.diffusion.rho
        sigma_min = self.diffusion.sigma_min
        sigma_max = self.diffusion.sigma_max

        noise_schedule = sigma_max ** (1 / rho) + indices / (self.diffusion.steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
        )
        return noise_schedule ** rho

    def validation_step(self, batch, batch_idx):
        images, cond_input = batch
        images = images.float()

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        # Compute validation loss
        loss = self._consistency_losses(z, cond_input)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
