import numpy as np
import torch
import torch.nn.functional as F
from piq import LPIPS
from utils.dist_utils import dev

from utils.nn_utils import mean_flat, append_dims, append_zero
from utils.random_utils import get_generator


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class KarrasDiffusion:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        steps=40,
        weight_schedule="karras",
        distillation=False,
        loss_norm="lpips",
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")
        self.rho = rho
        self.num_timesteps = steps

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, self.num_timesteps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        terms = {}

        sigmas_all = self.get_sigmas().to(dev())

        sigmas = sigmas_all[t]

        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        model_output, denoised = self.denoise(model, x_t, sigmas, **model_kwargs)

        snrs = self.get_snr(sigmas)
        weights = append_dims(
            get_weightings(self.weight_schedule, snrs, self.sigma_data), dims
        )
        terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
        terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
        return_latents_for_discriminator=False,
        rollout_steps=25,  # Added rollout_steps to support multiple unrolls
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)

        if target_model:
            @torch.no_grad()
            def target_denoise_fn(x, t):
                return self.denoise(target_model, x, t, **model_kwargs)
        else:
            raise NotImplementedError("Must have a target model")

        if teacher_model:
            @torch.no_grad()
            def teacher_denoise_fn(x, t):
                return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)

        @torch.no_grad()
        def heun_solver(x, t, next_t, x0, model_kwargs):
            # One step of Heun's method using teacher model
            model_output, denoised = teacher_denoise_fn(x, t)
            d = (x - denoised) / append_dims(t, dims)
            x_next = x + d * append_dims(next_t - t, dims)
            model_output, denoised_next = teacher_denoise_fn(x_next, next_t)
            d_2 = (x_next - denoised_next) / append_dims(next_t, dims)
            x_next = x + (d + d_2) * append_dims((next_t - t) / 2, dims)
            return x_next

        # Sample random timestep indices for each item in the batch
        if num_scales == 1:
            indices = torch.zeros(x_start.shape[0], device=x_start.device)
        else:
            indices = torch.randint(0, num_scales, (x_start.shape[0],), device=x_start.device)

        # Convert indices to actual time values t and t2
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        # Add noise to x_start to create x_t and x_t2_true
        x_t = x_start + noise * append_dims(t, dims)
        x_t2_true = x_start + noise * append_dims(t2, dims)

        # Store RNG state for reproducibility
        dropout_state = torch.get_rng_state()
        model_output, distiller = denoise_fn(x_t, t)

        torch.set_rng_state(dropout_state)
        model_output_target, distiller_target = target_denoise_fn(x_t2_true, t2)
        distiller_target = distiller_target.detach()

        # Compute weighted reconstruction loss
        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = torch.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(distiller_target, size=32, mode="bilinear")
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=224, mode="bilinear")
                distiller_target = F.interpolate(distiller_target, size=224, mode="bilinear")
            loss = self.lpips_loss((distiller + 1) / 2.0, (distiller_target + 1) / 2.0) * weights
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {"loss": loss}

        if return_latents_for_discriminator:
            x_fake = karras_sample(
                diffusion=self,
                model=model,
                shape=x_t.shape,
                steps=num_scales,
                model_kwargs=model_kwargs,
                device=x_t.device,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
            )

            x_consistency = x_fake.detach()

            # Additional outputs for adversarial training
            terms.update({
                "x_consistency": x_consistency,  # Generated latent after rollout
                "x_true": x_start,
                "model_kwargs": model_kwargs,
                # Timestep-based weighting (inverse ramp)
                "t_weight": 1.0 - ((t2 ** (1 / self.rho)) - (self.sigma_max ** (1 / self.rho))) /
                              ((self.sigma_min ** (1 / self.rho)) - (self.sigma_max ** (1 / self.rho))),
            })

        return terms


    # def consistency_losses(
    #     self,
    #     model,
    #     x_start,
    #     num_scales,
    #     model_kwargs=None,
    #     target_model=None,
    #     teacher_model=None,
    #     teacher_diffusion=None,
    #     noise=None,
    #     return_latents_for_discriminator=False,
    # ):
    #     if model_kwargs is None:
    #         model_kwargs = {}
    #     if noise is None:
    #         noise = torch.randn_like(x_start)
    #
    #     dims = x_start.ndim
    #
    #     def denoise_fn(x, t):
    #         return self.denoise(model, x, t, **model_kwargs)
    #
    #     if target_model:
    #
    #         @torch.no_grad()
    #         def target_denoise_fn(x, t):
    #             return self.denoise(target_model, x, t, **model_kwargs)
    #
    #     else:
    #         raise NotImplementedError("Must have a target model")
    #
    #     if teacher_model:
    #
    #         @torch.no_grad()
    #         def teacher_denoise_fn(x, t):
    #             return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)
    #
    #     @torch.no_grad()
    #     def heun_solver(samples, t, next_t, x0, model_kwargs):
    #         x = samples
    #         if teacher_model is None:
    #             denoiser = x0
    #         else:
    #             model_output, denoiser = teacher_denoise_fn(x, t)
    #         d = (x - denoiser) / append_dims(t, dims)
    #         samples = x + d * append_dims(next_t - t, dims)
    #         if teacher_model is None:
    #             denoiser = x0
    #         else:
    #             model_output, denoiser = teacher_denoise_fn(samples, next_t)
    #
    #         next_d = (samples - denoiser) / append_dims(next_t, dims)
    #         samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)
    #
    #         return samples
    #
    #     # @torch.no_grad()
    #     # def heun_solver(samples, t, next_t, x0):
    #     #     x = samples
    #     #     if teacher_model is None:
    #     #         denoiser = x0
    #     #     else:
    #     #         denoiser = teacher_denoise_fn(x, t)
    #     #
    #     #     d = denoiser  # Directly use it as difference
    #     #
    #     #     samples = x + d * append_dims(next_t - t, dims)
    #     #
    #     #     if teacher_model is None:
    #     #         denoiser = x0
    #     #     else:
    #     #         denoiser = teacher_denoise_fn(samples, next_t)
    #     #
    #     #     next_d = denoiser  # Same reasoning, just for next_t
    #     #
    #     #     samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)
    #     #
    #     #     return samples
    #
    #     @torch.no_grad()
    #     def euler_solver(samples, t, next_t, x0):
    #         x = samples
    #         if teacher_model is None:
    #             denoiser = x0
    #         else:
    #             model_output, denoiser = teacher_denoise_fn(x, t)
    #         d = (x - denoiser) / append_dims(t, dims)
    #         samples = x + d * append_dims(next_t - t, dims)
    #
    #         return samples
    #
    #     if num_scales == 1:
    #         indices = torch.zeros(x_start.shape[0], device=x_start.device)
    #     else:
    #         indices = torch.randint(
    #             0, num_scales - 1, (x_start.shape[0],), device=x_start.device
    #         )
    #
    #     t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
    #         self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
    #     )
    #     t = t**self.rho
    #
    #     t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
    #         self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
    #     )
    #     t2 = t2**self.rho
    #
    #     x_t = x_start + noise * append_dims(t, dims)
    #     x_t2_true = x_start + noise * append_dims(t2, dims)
    #
    #     dropout_state = torch.get_rng_state()
    #     model_output, distiller = denoise_fn(x_t, t)
    #
    #     if teacher_model is None:
    #         x_t2 = euler_solver(x_t, t, t2, x_start).detach()
    #     else:
    #         x_t2 = heun_solver(x_t, t, t2, x_start, model_kwargs).detach()
    #
    #     torch.set_rng_state(dropout_state)
    #     model_output_target, distiller_target = target_denoise_fn(x_t2, t2)
    #     distiller_target = distiller_target.detach()
    #
    #     ##########################################
    #
    #     def recursive_inference(x_start, noise, denoise_fn, target_denoise_fn, heun_solver, append_dims, model_kwargs):
    #         # Given time steps in descending order
    #         t_values = torch.tensor([10.0000, 5.6553, 3.0405, 1.5389, 0.7237, 0.3107, 0.1187, 0.0389, 0.0103, 0.002], device=x_start.device)
    #
    #         # Initialize x_t with noise
    #         x_t = x_start + noise * append_dims(t_values[0], dims)
    #
    #         for i in range(len(t_values) - 1):  # Iterate through time steps
    #             t, t2 = t_values[i], t_values[i + 1]  # Current and next time step
    #
    #             # Repeat t and t2 to match torch.Size([16])
    #             t = t.repeat(x_t.shape[0])
    #             t2 = t2.repeat(x_t.shape[0])
    #
    #             # Predict next state using Heun solver
    #             x_t2 = heun_solver(x_t, t, t2, x_start, model_kwargs).detach()
    #
    #             # Update x_t
    #             # x_t = target_denoise_fn(x_t2, t2).detach()  # Denoise at t2
    #             model_output, x_t = denoise_fn(x_t2, t2).detach()  # Denoise at t2
    #
    #         return x_t  # Return the final denoised output
    #
    #     ########################################
    #
    #     snrs = self.get_snr(t)
    #     weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
    #     if self.loss_norm == "l1":
    #         diffs = torch.abs(distiller - distiller_target)
    #         loss = mean_flat(diffs) * weights
    #     elif self.loss_norm == "l2":
    #         diffs = (distiller - distiller_target) ** 2
    #         loss = mean_flat(diffs) * weights
    #     elif self.loss_norm == "l2-32":
    #         distiller = F.interpolate(distiller, size=32, mode="bilinear")
    #         distiller_target = F.interpolate(
    #             distiller_target,
    #             size=32,
    #             mode="bilinear",
    #         )
    #         diffs = (distiller - distiller_target) ** 2
    #         loss = mean_flat(diffs) * weights
    #     elif self.loss_norm == "lpips":
    #         if x_start.shape[-1] < 256:
    #             distiller = F.interpolate(distiller, size=224, mode="bilinear")
    #             distiller_target = F.interpolate(
    #                 distiller_target, size=224, mode="bilinear"
    #             )
    #
    #         loss = (
    #             self.lpips_loss(
    #                 (distiller + 1) / 2.0,
    #                 (distiller_target + 1) / 2.0,
    #             )
    #             * weights
    #         )
    #     else:
    #         raise ValueError(f"Unknown loss norm {self.loss_norm}")
    #
    #     terms = {}
    #     terms["loss"] = loss
    #
    #     # Optionally return x_t2_hat and x_t2 for adversarial loss
    #     if return_latents_for_discriminator:
    #         # Heun's metorchod for next timestep prediction
    #         dt = t2 - t
    #         _, denoised = denoise_fn(x_t, t)
    #         d = to_d(x_t, t, denoised)
    #         x_t2 = x_t + d * append_dims(dt, dims)
    #         _, denoised_2 = denoise_fn(x_t2, t2)
    #         d_2 = to_d(x_t2, t2, denoised_2)
    #         d_prime = (d + d_2) / 2
    #         x_t2_consistency = x_t + d_prime * append_dims(dt, dims)
    #
    #         terms["x_t2_consistency"] = x_t2_consistency
    #         terms["x_t2_model_output"] = model_output
    #         terms["x_t2_target"] = distiller_target
    #         terms["x_t2_target_model_output"] = model_output_target
    #         terms["x_t2_true"] = x_t2_true
    #         terms["t2"] = t2
    #         terms["x_start"] = x_start
    #         terms["model_kwargs"] = model_kwargs
    #         # Timestep-based weighting (inverse ramp)
    #         terms["t_weight"] = 1.0 - ((t2 ** (1 / self.rho)) - (self.sigma_max ** (1 / self.rho))) / \
    #                             ((self.sigma_min ** (1 / self.rho)) - (self.sigma_max ** (1 / self.rho)))
    #
    #     return terms

    # def consistency_losses(
    #         self,
    #         model,
    #         x_start,
    #         num_scales,
    #         model_kwargs=None,
    #         target_model=None,
    #         teacher_model=None,
    #         teacher_diffusion=None,
    #         noise=None,
    # ):
    #     if model_kwargs is None:
    #         model_kwargs = {}
    #     if noise is None:
    #         noise = torch.randn_like(x_start)
    #
    #     dims = x_start.ndim
    #
    #     def denoise_fn(x, t):
    #         return self.denoise(model, x, t, **model_kwargs)[1]
    #
    #     if target_model:
    #
    #         @torch.no_grad()
    #         def target_denoise_fn(x, t):
    #             return self.denoise(target_model, x, t, **model_kwargs)[1]
    #
    #     else:
    #         raise NotImplementedError("Must have a target model")
    #
    #     if teacher_model:
    #         @torch.no_grad()
    #         def teacher_denoise_fn(x, t):
    #             return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]
    #
    #     @torch.no_grad()
    #     def heun_solver(samples, t, next_t, x0):
    #         x = samples
    #         if teacher_model is None:
    #             denoiser = x0
    #         else:
    #             denoiser = teacher_denoise_fn(x, t)
    #         d = (x - denoiser) / append_dims(t, dims)
    #         samples = x + d * append_dims(next_t - t, dims)
    #         if teacher_model is None:
    #             denoiser = x0
    #         else:
    #             denoiser = teacher_denoise_fn(samples, next_t)
    #
    #         next_d = (samples - denoiser) / append_dims(next_t, dims)
    #         samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)
    #
    #         return samples
    #
    #     @torch.no_grad()
    #     def euler_solver(samples, t, next_t, x0):
    #         x = samples
    #         if teacher_model is None:
    #             denoiser = x0
    #         else:
    #             denoiser = teacher_denoise_fn(x, t)
    #         d = (x - denoiser) / append_dims(t, dims)
    #         samples = x + d * append_dims(next_t - t, dims)
    #
    #         return samples
    #
    #     indices = torch.randint(
    #         0, num_scales - 1, (x_start.shape[0],), device=x_start.device
    #     )
    #
    #     t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
    #             self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
    #     )
    #     t = t ** self.rho
    #
    #     t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
    #             self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
    #     )
    #     t2 = t2 ** self.rho
    #
    #     x_t = x_start + noise * append_dims(t, dims)
    #
    #     dropout_state = torch.get_rng_state()
    #     distiller = denoise_fn(x_t, t)
    #
    #     if teacher_model is None:
    #         x_t2 = euler_solver(x_t, t, t2, x_start).detach()
    #     else:
    #         x_t2_euler = euler_solver(x_t, t, t2, x_start).detach()
    #         x_t2 = heun_solver(x_t, t, t2, x_start).detach()
    #
    #     torch.set_rng_state(dropout_state)
    #     distiller_target = target_denoise_fn(x_t2, t2)
    #     distiller_target = distiller_target.detach()
    #
    #     # Convert to numpy arrays
    #     x_euler_np = x_t2_euler.cpu().numpy()
    #     x_heun_np = x_t2.cpu().numpy()
    #     t_np = t.cpu().numpy()
    #
    #     # Ensure `t_np` contains unique values
    #     t_np, unique_indices = np.unique(t_np, return_index=True)
    #     x_euler_np = x_euler_np[unique_indices]
    #     x_heun_np = x_heun_np[unique_indices]
    #
    #     from scipy.interpolate import interp1d
    #     import matplotlib.pyplot as plt
    #
    #     # Interpolate results for smoother curves
    #     # interp_func_euler = interp1d(t_np, x_euler_np, kind="cubic", axis=0)
    #     # interp_func_heun = interp1d(t_np, x_heun_np, kind="cubic", axis=0)
    #     # t_high_res = np.linspace(t_np.min(), t_np.max(), 5000)
    #     # x_euler_high_res = interp_func_euler(t_high_res)
    #     # x_heun_high_res = interp_func_heun(t_high_res)
    #
    #     x_euler_high_res = x_euler_np
    #     x_heun_high_res = x_heun_np
    #     t_high_res = t_np
    #
    #     # Compute divergence
    #     divergence = np.abs(x_euler_high_res - x_heun_high_res)  # Shape: (5000, 1, 32)
    #
    #     # Reduce dimensions for plotting
    #     divergence_mean = divergence.mean(axis=(-1, -2))  # Average over last two axes
    #     # Shape: (5000,)
    #
    #     # Plot the divergence
    #     plt.figure(figsize=(12, 6))
    #
    #     # First plot: Divergence between Euler and Heun
    #     plt.subplot(1, 2, 1)
    #     plt.plot(t_high_res, divergence_mean, label="Divergence (Mean)")
    #     plt.yscale("log")
    #     plt.xlabel("Time")
    #     plt.ylabel("Divergence (log scale)")
    #     plt.title("Divergence Between Solvers")
    #     plt.legend()
    #
    #     # Second plot: Euler and Heun method values
    #     plt.subplot(1, 2, 2)
    #     x_euler_mean = x_euler_high_res.mean(axis=(-1, -2))  # Average over last two axes
    #     x_heun_mean = x_heun_high_res.mean(axis=(-1, -2))  # Average over last two axes
    #
    #     plt.plot(t_high_res, x_euler_mean, label="Euler (Mean)", linestyle="--")
    #     plt.plot(t_high_res, x_heun_mean, label="Heun (Mean)", linestyle="-")
    #     plt.xlabel("Time")
    #     plt.ylabel("Values")
    #     plt.title("Euler vs Heun Values")
    #     plt.legend()
    #
    #     # Adjust layout and show the plots
    #     plt.tight_layout()
    #     plt.show()
    #
    #     def compute_jacobian(f, x, t, model_kwargs):
    #         """
    #         Compute the Jacobian matrix J = ∂f/∂x numerically using finite differences.
    #
    #         Args:
    #             f: Function representing the system dynamics (e.g., denoise_fn)
    #             x: Tensor of the current state, shape (batch_size, ...)
    #             t: Time, scalar or tensor
    #             model_kwargs: Additional arguments for f
    #
    #         Returns:
    #             J: Jacobian matrix of shape (batch_size, x_dim, x_dim)
    #         """
    #         epsilon = 1e-5  # Small perturbation
    #         batch_size = x.size(0)
    #         x_dim = x.view(batch_size, -1).size(1)  # Flatten spatial dimensions if needed
    #         J = torch.zeros(batch_size, x_dim, x_dim, device=x.device)
    #
    #         # Perturb each dimension of x
    #         for i in range(x_dim):
    #             x_perturb = x.clone().view(batch_size, -1)
    #             x_perturb[:, i] += epsilon
    #             f_perturb = f(x_perturb.view_as(x), t)  # Reshape back if necessary
    #             f_original = f(x, t)
    #             J[:, :, i] = ((f_perturb - f_original) / epsilon).view(batch_size, -1)
    #
    #         return J
    #
    #     def compute_stiffness_ratio(J):
    #         """
    #         Compute the stiffness ratio R_stiffness = |λ_max| / |λ_min|.
    #
    #         Args:
    #             J: Jacobian matrix of shape (batch_size, x_dim, x_dim)
    #
    #         Returns:
    #             R_stiffness: Tensor of stiffness ratios for each batch element
    #         """
    #         batch_size = J.size(0)
    #         stiffness_ratios = torch.zeros(batch_size, device=J.device)
    #
    #         for i in range(batch_size):
    #             eigenvalues = torch.linalg.eigvals(J[i])  # Eigenvalues of the i-th Jacobian
    #             max_eigenvalue = torch.max(torch.abs(eigenvalues))
    #             min_eigenvalue = torch.min(torch.abs(eigenvalues))
    #             stiffness_ratios[i] = max_eigenvalue / min_eigenvalue
    #
    #         return stiffness_ratios
    #
    #     # Define a wrapper around your denoise function
    #     def system_dynamics(x, t):
    #         return denoise_fn(x, t)  # Use the denoise_fn defined in consistency_losses
    #
    #     # Example: Calculate Jacobian and stiffness ratio
    #     x_example = x_start.clone()  # Replace with a specific batch example
    #     t_example = t  # Replace with specific time point
    #     J = compute_jacobian(system_dynamics, x_example, t_example, model_kwargs)
    #     R_stiffness = compute_stiffness_ratio(J)
    #
    #     # Print results
    #     print("Jacobian matrix:\n", J)
    #     print("Stiffness ratios:\n", R_stiffness)
    #
    #     breakpoint()
    #
    #     # Compute the loss
    #     snrs = self.get_snr(t)
    #     weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
    #     if self.loss_norm == "l1":
    #         diffs = torch.abs(distiller - distiller_target)
    #         loss = mean_flat(diffs) * weights
    #     elif self.loss_norm == "l2":
    #         diffs = (distiller - distiller_target) ** 2
    #         loss = mean_flat(diffs) * weights
    #     elif self.loss_norm == "l2-32":
    #         distiller = F.interpolate(distiller, size=32, mode="bilinear")
    #         distiller_target = F.interpolate(
    #             distiller_target,
    #             size=32,
    #             mode="bilinear",
    #         )
    #         diffs = (distiller - distiller_target) ** 2
    #         loss = mean_flat(diffs) * weights
    #     elif self.loss_norm == "lpips":
    #         if x_start.shape[-1] < 256:
    #             distiller = F.interpolate(distiller, size=224, mode="bilinear")
    #             distiller_target = F.interpolate(
    #                 distiller_target, size=224, mode="bilinear"
    #             )
    #
    #         loss = (
    #                 self.lpips_loss(
    #                     (distiller + 1) / 2.0,
    #                     (distiller_target + 1) / 2.0,
    #                 )
    #                 * weights
    #         )
    #     else:
    #         raise ValueError(f"Unknown loss norm {self.loss_norm}")
    #
    #     terms = {}
    #     terms["loss"] = loss
    #
    #     return terms

    def progdist_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        @torch.no_grad()
        def teacher_denoise_fn(x, t):
            return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @torch.no_grad()
        def euler_solver(samples, t, next_t):
            x = samples
            denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)

            return samples

        @torch.no_grad()
        def euler_to_denoiser(x_t, t, x_next_t, next_t):
            denoiser = x_t - append_dims(t, dims) * (x_next_t - x_t) / append_dims(
                next_t - t, dims
            )
            return denoiser

        indices = torch.randint(0, num_scales, (x_start.shape[0],), device=x_start.device)

        t = self.sigma_max ** (1 / self.rho) + indices / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 0.5) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        t3 = self.sigma_max ** (1 / self.rho) + (indices + 1) / num_scales * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t3 = t3**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        denoised_x = denoise_fn(x_t, t)

        x_t2 = euler_solver(x_t, t, t2).detach()
        x_t3 = euler_solver(x_t2, t2, t3).detach()

        target_x = euler_to_denoiser(x_t, t, x_t3, t3).detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = torch.abs(denoised_x - target_x)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (denoised_x - target_x) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                denoised_x = F.interpolate(denoised_x, size=224, mode="bilinear")
                target_x = F.interpolate(target_x, size=224, mode="bilinear")
            loss = (
                self.lpips_loss(
                    (denoised_x + 1) / 2.0,
                    (target_x + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        terms = {}
        terms["loss"] = loss

        return terms

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        if not self.distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, model_kwargs)

        # --- 对齐 model_output 与 x_t 的 3D 空间尺寸 ---
        if model_output.shape != x_t.shape:
            min_d = min(model_output.shape[-3], x_t.shape[-3])
            min_h = min(model_output.shape[-2], x_t.shape[-2])
            min_w = min(model_output.shape[-1], x_t.shape[-1])
            model_output = model_output[..., :min_d, :min_h, :min_w]
            x_t          = x_t[...,          :min_d, :min_h, :min_w]
        # --------------------------------------------------

        denoised = c_out * model_output + c_skip * x_t

        return model_output, denoised


def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80.0,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
):
    if generator is None:
        generator = get_generator("dummy")

    if sampler == "progdist":
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    x_T = generator.randn(*shape, device=device) * sigma_max

    sample_fn = {
        "heun": sample_heun,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise, model_kwargs=model_kwargs
        )
    elif sampler == "multistep":
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=diffusion.rho, steps=steps
        )
    else:
        sampler_args = {}

    def denoiser(x_t, sigma):
        _, denoised = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    return x_0.clamp(-1, 1)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="None"):
    """Constructs torch noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    # --- 对齐 x 与 denoised 的 3D 空间尺寸 ---
    if x.shape != denoised.shape:
        min_d = min(x.shape[-3], denoised.shape[-3])
        min_h = min(x.shape[-2], denoised.shape[-2])
        min_w = min(x.shape[-1], denoised.shape[-1])
        x        = x[...,        :min_d, :min_h, :min_w]
        denoised = denoised[..., :min_d, :min_h, :min_w]
    # ------------------------------------------
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates torche noise level (sigma_down) to step down to and torche amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling witorch Euler metorchod steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler metorchod
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@torch.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling witorch midpoint metorchod steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x


@torch.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    model_kwargs=None,
    currentepoch=0,

):
    """Implements Algoritorchm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # if currentepoch > 50 and i == indices[-2]:
        #     import matplotlib.pyplot as plt
        #     fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        #
        #     # Plot the image at index 0 on the left
        #     axes[0,0].imshow(x[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
        #     axes[0,0].set_title(model_kwargs['tensor'][0][4:6])
        #     axes[0,0].axis("off")
        #
        #     # Plot the image at index 1 on the right
        #     axes[0,1].imshow(x[1, 0, :, :].detach().cpu().numpy(), cmap='gray')
        #     axes[0,1].set_title(model_kwargs['tensor'][1][4:6])
        #     axes[0,1].axis("off")
        #
        #     # Plot the image at index 0 on the left
        #     axes[1,0].imshow(x[2, 0, :, :].detach().cpu().numpy(), cmap='gray')
        #     axes[1,0].set_title(model_kwargs['tensor'][2][4:6])
        #     axes[1,0].axis("off")
        #
        #     # Plot the image at index 1 on the right
        #     axes[1,1].imshow(x[3, 0, :, :].detach().cpu().numpy(), cmap='gray')
        #     axes[1,1].set_title(model_kwargs['tensor'][3][4:6])
        #     axes[1,1].axis("off")
        #
        #     plt.tight_layout()
        #     plt.show()
        #     breakpoint()

        # --- 安全对齐：防止 25 vs 24 尺寸不匹配 ---
        def align(a, b):
            """确保两个张量最后三维一致"""
            min_d = min(a.shape[-3], b.shape[-3])
            min_h = min(a.shape[-2], b.shape[-2])
            min_w = min(a.shape[-1], b.shape[-1])
            return a[..., :min_d, :min_h, :min_w], b[..., :min_d, :min_h, :min_w]
        # ---------------------------------------------------

        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler metorchod
            x, d = align(x, d) #对齐
            x = x + d * dt
        else:
            # Heun's metorchod
            x, d = align(x, d) #对齐
            x_2 = x + d * dt

            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            
            d, d_2 = align(d, d_2)   # 再对齐，确保 d 与 d_2 一致
            d_prime = (d + d_2) / 2
            
            x, d_prime = align(x, d_prime)   # 最后对齐一次，防止 x 与 d_prime 形状不匹配
            x = x + d_prime * dt

    return x


@torch.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
):
    """Implements Algoritorchm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@torch.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algoritorchm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint metorchod, where torche midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@torch.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@torch.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip torche zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x


@torch.no_grad()
def iterative_colorization(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    def obtain_ortorchogonal_matrix():
        vector = np.asarray([0.2989, 0.5870, 0.1140])
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(3)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = torch.from_numpy(obtain_ortorchogonal_matrix()).to(dev()).to(torch.float32)
    mask = torch.zeros(*x.shape[1:], device=dev())
    mask[0, ...] = 1.0

    def replacement(x0, x1):
        x0 = torch.einsum("bchw,cd->bdhw", x0, Q)
        x1 = torch.einsum("bchw,cd->bdhw", x1, Q)

        x_mix = x0 * mask + x1 * (1.0 - mask)
        x_mix = torch.einsum("bdhw,cd->bchw", x_mix, Q)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, torch.zeros_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = torch.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@torch.no_grad()
def iterative_inpainting(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    from PIL import Image, ImageDraw, ImageFont

    image_size = x.shape[-1]

    # create a blank image witorch a white background
    img = Image.new("RGB", (image_size, image_size), color="white")

    # get a drawing context for torche image
    draw = ImageDraw.Draw(img)

    # load a font
    font = ImageFont.truetype("arial.ttf", 250)

    # draw torche letter "C" in black
    draw.text((50, 0), "S", font=font, fill=(0, 0, 0))

    # convert torche image to a numpy array
    img_np = np.array(img)
    img_np = img_np.transpose(2, 0, 1)
    img_torch = torch.from_numpy(img_np).to(dev())

    mask = torch.zeros(*x.shape, device=dev())
    mask = mask.reshape(-1, 7, 3, image_size, image_size)

    mask[::2, :, img_torch > 0.5] = 1.0
    mask[1::2, :, img_torch < 0.5] = 1.0
    mask = mask.reshape(-1, 3, image_size, image_size)

    def replacement(x0, x1):
        x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = replacement(images, -torch.ones_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = torch.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images


@torch.no_grad()
def iterative_superres(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
):
    patch_size = 8

    def obtain_ortorchogonal_matrix():
        vector = np.asarray([1] * patch_size**2)
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(patch_size**2)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = torch.from_numpy(obtain_ortorchogonal_matrix()).to(dev()).to(torch.float32)

    image_size = x.shape[-1]

    def replacement(x0, x1):
        x0_flatten = (
            x0.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x1_flatten = (
            x1.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x0 = torch.einsum("bcnd,de->bcne", x0_flatten, Q)
        x1 = torch.einsum("bcnd,de->bcne", x1_flatten, Q)
        x_mix = x0.new_zeros(x0.shape)
        x_mix[..., 0] = x0[..., 0]
        x_mix[..., 1:] = x1[..., 1:]
        x_mix = torch.einsum("bcne,de->bcnd", x_mix, Q)
        x_mix = (
            x_mix.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )
        return x_mix

    def average_image_patches(x):
        x_flatten = (
            x.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
        return (
            x_flatten.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    images = average_image_patches(images)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        x0 = torch.clamp(x0, -1.0, 1.0)
        x0 = replacement(images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x, images
