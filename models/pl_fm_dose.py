import os
from datetime import datetime
import pytorch_lightning as pl
import torch
import yaml
import torch.nn as nn
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import numpy as np
from typing import Optional, Dict, Any
from torchmetrics.image import StructuralSimilarityIndexMeasure  # Import SSIM metric
from utils.evaluation_tools import ImageSimilarityMetrics
from utils.resample import LossAwareSampler, UniformSampler
from torch_ema import ExponentialMovingAverage
#from denoiser.karras_denoiser import karras_sample


class LightningLatentFlowMatching(pl.LightningModule):
    def __init__(
            self,
            unet_model: nn.Module,
            vae_model: nn.Module,
            learning_rate: float,
            num_steps: int,
            plot_example_images_epoch_start: int,
            weight_decay=0.0,
            vae_down_sample=4,

            ode_solver="euler"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['unet_model', 'vae_model'])

        # Models
        self.model = unet_model
        self.vae = vae_model
        self.num_steps = num_steps
        self.ode_solver = ode_solver

        # Training params
        self.plot_example_images_epoch_start = plot_example_images_epoch_start
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.vae_down_sample = vae_down_sample

        # Loss function and metrics
        self.criterion_mse = nn.MSELoss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

        # Metrics
        self.similarity_calculator = ImageSimilarityMetrics()

        # Ensure UNet parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True

        self.similarity_calculator = ImageSimilarityMetrics()

        ###############
        # MANUAL OPTIM CODE (to show it performs the same as the automatic including grad clipping
        self.automatic_optimization = False  # Set manual optimization
        self.scaler = torch.cuda.amp.GradScaler()  # Use GradScaler for mixed precision training
        self.optimizer = self.configure_optimizers()
        self.best_val_loss = float("inf")  # Track the best validation loss
        self.start_time = datetime.now().strftime("%d_%m_%Y-%H-%M")  # Store the training start time once
        ################

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.99)


    def forward(self, noisy_latents: torch.Tensor, t: torch.Tensor,
                cond_input: Optional[Dict[str, torch.Tensor]] = None):
        return self.model(noisy_latents, t, cond_input)

    def training_step(self, batch, batch_idx, logging=True):
        images, cond_input = batch
        images = images.float()

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        #t = torch.rand(images.shape[0], device=self.device)
        t = torch.rand(z.shape[0], device=z.device)
        #t = torch.distributions.Beta(0.5, 0.5).sample((images.shape[0],)).to(self.device)

        z0 = torch.randn_like(z)
        t_view = t.view(z.shape[0], *([1] * (z.dim()-1)))
        z_t = (1 - t_view) * z0 + t_view * z
        v_target = z - z0
        v_pred = self.forward(z_t, t, cond_input)
        loss = self.criterion_mse(v_pred, v_target)


        if logging:
            self.log("train/loss", loss, prog_bar=True)

        # ✅ Zero gradients before backward pass
        self.optimizer.zero_grad()

        # ✅ Compute scaled gradients using GradScaler
        self.scaler.scale(loss).backward()

        # ✅ Unscale gradients before clipping (MUST DO THIS IN MANUAL OPTIMIZATION!)
        self.scaler.unscale_(self.optimizer)

        # ✅ Apply manual gradient clipping (Same as Trainer's `gradient_clip_val=1.0`)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # ✅ Perform optimizer step with GradScaler
        self.scaler.step(self.optimizer)
        self.scaler.update()  # Updates scaling factor for next step

        self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        images, cond_input = batch
        images = images.float()

        # Store original weights
        self.ema.store(self.model.parameters())

        # Copy EMA weights to the model
        self.ema.copy_to(self.model.parameters())

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        B = z.shape[0]
        #t = torch.rand(B, device=self.device)
        t = torch.rand(B, device=z.device)
        z0 = torch.randn_like(z)   # noise
        t_view = t.view(B, *([1] * (z.dim() - 1)))
        z_t = (1 - t_view) * z0 + t_view * z  # noise -> data

        v_target = z - z0
        v_pred = self.forward(z_t, t, cond_input)
        loss = self.criterion_mse(v_pred, v_target)

        # Log validation loss
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # Restore original weights
        self.ema.restore(self.model.parameters())

        return loss

    def test_step(self, batch, batch_idx):
        images, cond_input = batch
        images = images.float()

        # Encode images to latent space
        with torch.no_grad():
            z, _ = self.vae.encode(images, None)

        # Generate samples using EMA weights
        _ = self.generate_samples(images, z, cond_input, use_ema=True, progress=True, save_images=True)

    def on_train_epoch_end(self):
        if self.current_epoch < self.plot_example_images_epoch_start:
            return

        val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
        images, cond_input = val_batch
        images = images.float().to(self.device)

        with torch.no_grad():
            im, _ = self.vae.encode(images, None)
            _ = self.generate_samples(images, z=im, cond_input=cond_input, use_ema=True)

    def on_validation_end(self) -> None:
        if not self.automatic_optimization:
            """Manually save the best checkpoint when val_loss improves."""
            val_loss = self.trainer.callback_metrics.get("val/loss", None)

            if val_loss is None:
                print("⚠️ val/loss not found in callback metrics!")
                return

            val_loss = val_loss.item()  # Convert from tensor to float

            # Check if the new validation loss is better
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss  # Update best loss

                # Use the stored start_time
                checkpoint_path = os.path.join(self.trainer.default_root_dir, "./checkpoints/fm/",
                                               f"best_model_{self.start_time}.ckpt")
                print(f"✅ New best model found! Saving checkpoint to {checkpoint_path}")
                self.trainer.save_checkpoint(checkpoint_path)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)  # Ensure compatibility with PyTorch's state_dict
        if hasattr(self.ema, "state_dict") and callable(getattr(self.ema, "state_dict")):
            state["ema"] = self.ema.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        if "ema" in state_dict:
            self.ema.load_state_dict(state_dict["ema"])  # Load EMA weights
            print("✅ EMA weights restored from checkpoint!")
        else:
            print("⚠️ No EMA weights found in checkpoint!")

        # Remove "ema" from state_dict to avoid conflicts with strict loading
        state_dict = {k: v for k, v in state_dict.items() if k != "ema"}

        super().load_state_dict(state_dict, strict=strict, *args, **kwargs)


    def generate_samples(self, images, z, cond_input=None, use_ema=False, progress=False, save_images=False):
        """
        Generate samples using the NoiseSampler.
        Args:
            images (torch.tensor): original images
            z (torch.tensor): latent space tensor (e.g., (channels, height, width)).
            cond_input (dict): Conditioning inputs for the model.
            use_ema (bool): Whether to use EMA weights for sampling.
            progress (bool): Whether to show the progress bar when sampling
            save_images (bool): Whether to save the generated images (should only be used during testing)

        Returns:
            torch.Tensor: Decoded samples from the VAE.
        """
        was_training = self.training
        self.eval()  # Ensure the model is in evaluation mode

        if use_ema:
            # Temporarily load EMA weights
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())


        try:
            with torch.no_grad():
                z_data = z
                z = torch.randn_like(z_data)  # same shape as input latent z
                N = self.num_steps
                t_grid = torch.linspace(0.0, 1.0, N + 1, device=z.device)
                for i in range(N):
                    t_i = t_grid[i].expand(z.shape[0])          # shape (B,)
                    dt = t_grid[i+1] - t_grid[i]               # positive
                    v = self.forward(z, t_i, cond_input)        # v(z,t)
                    z = z + dt * v
                    #t_i = t_grid[i].expand(z.shape[0])
                    #t_next = t_grid[i + 1].expand(z.shape[0])
                    #dt = t_grid[i + 1] - t_grid[i]
                    #v1 = self.forward(z, t_i, cond_input)
                    #z_euler = z + dt * v1
                    #v2 = self.forward(z_euler, t_next, cond_input)
                    #z = z + 0.5 * dt * (v1 + v2)
                
                generated_latents = z
                generated_images = self.vae.decode(generated_latents, None, images.shape)
        finally:
            if use_ema:
                # Restore original weights
                self.ema.restore(self.model.parameters())
            if was_training:
                self.train()

        # Calculate all metrics using the calculate_all_metrics() method
        mean_mse, mse_per_image = self.similarity_calculator.calculate_batch_mse(images, generated_images)
        mean_ssim, ssim_per_image = self.similarity_calculator.calculate_ssim(images, generated_images)
        mean_psnr, psnr_per_image = self.similarity_calculator.calculate_psnr(images, generated_images)
        mean_lpips, lpips_per_image = self.similarity_calculator.calculate_lpips(images, generated_images)
        mean_snr, snr_per_image = self.similarity_calculator.calculate_snr(images, generated_images)

        # Log mean metrics to WandB
        wandb.log({
            'fm_epoch': self.current_epoch,
            'fm_mse_mean': mean_mse,
            'fm_ssim_mean': mean_ssim,
            'fm_psnr_mean': mean_psnr,
            'fm_lpips_mean': mean_lpips,
            'fm_snr_mean': mean_snr,
        })

        num_samples = 2
        # Visualize and log the generated images
        ct_batch = cond_input["ct"].detach().cpu()
        fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3 * num_samples))
        #fig, axes = plt.subplots(num_samples, 3, figsize=(5*num_samples+1, 15))
        for i in range(num_samples):
            # Take center slice along depth dimension (D)
            mid_slice = images.shape[2] // 2
            orig_slice = images[i, 0, mid_slice, :, :].detach().cpu().numpy()
            true_latent_slice = z_data[i, 0, mid_slice // self.vae_down_sample, :, :].detach().cpu().numpy()
            generated_latent_slice = generated_latents[i, 0, mid_slice // self.vae_down_sample, :, :].detach().cpu().numpy()
            recon_slice = generated_images[i, 0, mid_slice, :, :].detach().cpu().numpy()
            #orig_scaled  = orig_slice  *  10.0
            #recon_scaled = recon_slice * 10.0
            ct_slice = ct_batch[i, 0, mid_slice, :, :]


            vmin = 0.0
            vmax = np.percentile(orig_slice, 99.5)

            axes[i, 0].imshow(orig_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, 0].set_title("Original image")
            axes[i, 1].imshow(recon_slice, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, 1].set_title("Reconstructed image")

            axes[i, 2].imshow(true_latent_slice, cmap="gray")
            axes[i, 2].set_title("True latent")
            axes[i, 3].imshow(generated_latent_slice, cmap="gray")
            axes[i, 3].set_title("Generated latent")
            
            axes[i, 4].imshow(ct_slice, cmap="gray")
            axes[i, 4].set_title("CT Input")

            for j in range(5):
                axes[i, j].axis("off")

        plt.tight_layout()
        self.logger.experiment.log({f"FM_outputs_epoch_{self.current_epoch}": wandb.Image(fig)})
        plt.close(fig)

        # Log the figure
        self.logger.log_image(
            key="FM_generated_images_epoch",
            images=[wandb.Image(fig)],
            step=self.current_epoch
        )
        plt.close(fig)

        if save_images:
            #########################################
            # save images

            # Create output directory if it doesn't exist
            output_dir = os.path.join("sample_outputs/", f"dim_{images.shape[-1]}/")
            os.makedirs(output_dir, exist_ok=True)

            # Save images as individual .npy files
            for i in range(images.shape[0]):
                # Save original image
                orig_img_np = images[i].detach().cpu().numpy()
                np.save(os.path.join(output_dir, f"original_{i}.npy"), orig_img_np)

                # Save generated image
                gen_img_np = generated_images[i].detach().cpu().numpy()
                np.save(os.path.join(output_dir, f"generated_{i}.npy"), gen_img_np)

                # Save CT image from condition if it exists
                if cond_input and "ct" in cond_input:
                    ct_img_np = cond_input["ct"][i].detach().cpu().numpy()
                    np.save(os.path.join(output_dir, f"ct_{i}.npy"), ct_img_np)

            # Save other conditioning inputs (non-CT) as YAML
            if cond_input:
                cond_yaml_data = {}

                for i in range(images.shape[0]):
                    cond_yaml_data[i] = {}
                    for key, value in cond_input.items():
                        if key != "ct":
                            cond_yaml_data[i][key] = value[i].detach().cpu().numpy().tolist()

                yaml_path = os.path.join(output_dir, "conditions.yaml")
                with open(yaml_path, "w") as f:
                    yaml.dump(cond_yaml_data, f, sort_keys=False)

        return generated_images

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer