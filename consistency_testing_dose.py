import yaml
import os
import pytorch_lightning as pl
import argparse
import torch
from pytorch_lightning.loggers import WandbLogger
from models.pl_consistency_model import LightningConsistencyModel
from models.pl_ldm_dose import LightningLatentDiffusion
from models.pl_vqvae_dose import LightningVQVAE  # Your VAE implementation
from models.unet_cond import UnetDose as Unet
from denoiser.karras_denoiser import KarrasDiffusion  # Updated Karras denoiser
from dataset.pl_dose_dataset import DoseDataModule
from utils.resample import create_named_schedule_sampler


def test(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    diffusion_model_config = config['ldm_params']
    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    test_config = config['train_params']
    consistency_config = config['consistency_params']

    # Initialize the test data module
    data_module = DoseDataModule(
        root_dir=dataset_config['im_path'],  # use this to point to /hdd/Josch_Data/simulations
        batch_size=test_config['ldm_batch_size'],
        num_workers=0,
        split=(0.8, 0.1, 0.1),
        downsample_factor=dataset_config['downsample_factor']
    )


    # Load the checkpoint if specified
    vqvae_checkpoint_path = test_config.get('vqvae_autoencoder_ckpt_name', None)
    if vqvae_checkpoint_path:
        vae = LightningVQVAE.load_from_checkpoint(
            vqvae_checkpoint_path,
            im_channels=dataset_config['im_channels'],
            model_config=autoencoder_model_config
        ).to(device)
        vae.eval()
    else:
        raise ValueError("VAE path must be specified for latent diffusion setup!")

    # Initialize models and scheduler
    consistency_model_config = diffusion_model_config
    consistency_model_config['dropout_rate'] = 0.0
    unet_consistency = Unet(autoencoder_model_config['z_channels'], consistency_model_config).to(device)
    unet_consistency.eval()

    # Initialize models and scheduler
    unet_target = Unet(autoencoder_model_config['z_channels'], consistency_model_config).to(device)
    unet_target.eval()

    """Load the LDM checkpoint if specified"""
    # Initialize models and scheduler
    unet_teacher = Unet(autoencoder_model_config['z_channels'], diffusion_model_config).to(device)
    unet_teacher.eval()

    schedule_sampler = create_named_schedule_sampler('uniform', diffusion_config["num_timesteps"])

    # Initialize noise scheduler
    karras_diffusion_teacher = KarrasDiffusion(
        sigma_min=diffusion_config["sigma_min"],
        sigma_max=diffusion_config["sigma_max"],
        rho=diffusion_config["rho"],
        distillation=False,
        steps=diffusion_config["num_timesteps"],
    )

    # Initialize noise scheduler
    karras_diffusion_consistency = KarrasDiffusion(
        sigma_min=diffusion_config["sigma_min"],
        sigma_max=diffusion_config["sigma_max"],
        rho=diffusion_config["rho"],
        distillation=True,
        steps=consistency_config["num_scales"],  # 1
        loss_norm=consistency_config["loss_norm"]
    )

    ldm_checkpoint_path = test_config.get('ldm_ckpt_cond_name', None)
    ldm = LightningLatentDiffusion.load_from_checkpoint(
        ldm_checkpoint_path,
        unet_model=unet_teacher,
        vae_model=vae,
        diffusion=karras_diffusion_teacher,
        learning_rate=test_config['ldm_lr'],
        plot_example_images_epoch_start=30
    ).to(device)
    ldm.eval()

    # Initialize LightningConsistencyModel
    consistency_checkpoint_path = test_config.get('consistency_ckpt_cond_name', None)
    if consistency_checkpoint_path is not None:
        consistency_model = LightningConsistencyModel.load_from_checkpoint(
            consistency_checkpoint_path,
            model=unet_consistency,
            diffusion=karras_diffusion_consistency,
            vae_model=vae,
            teacher_model=ldm.model,
            target_model=unet_target,
            schedule_sampler=schedule_sampler,
            teacher_diffusion=karras_diffusion_teacher,
            training_mode="consistency_distillation",
            plot_example_images_epoch_start=10,
            lr=test_config['consistency_lr'],
            num_scales=consistency_config["num_scales"],
            weight_decay=consistency_config["weight_decay"],
            ema_decay=consistency_config["ema_decay"]
        )
        # Manually restore EMA weights from the checkpoint
        checkpoint = torch.load(consistency_checkpoint_path, map_location=device)

        if "ema" in checkpoint["state_dict"]:
            consistency_model.ema.load_state_dict(checkpoint["state_dict"]["ema"])
            print("✅ EMA weights restored from checkpoint!")
        else:
            print("⚠️ Warning: No EMA weights found in checkpoint!")

    else:
        consistency_model = LightningConsistencyModel(
            model=unet_consistency,
            diffusion=karras_diffusion_consistency,
            vae_model=vae,
            teacher_model=ldm.model,
            target_model=unet_target,
            schedule_sampler=schedule_sampler,
            teacher_diffusion=karras_diffusion_teacher,
            training_mode="consistency_distillation",
            plot_example_images_epoch_start=10,
            lr=test_config['consistency_lr'],
            num_scales=consistency_config["num_scales"],
            weight_decay=consistency_config["weight_decay"],
            ema_decay=consistency_config["ema_decay"]
        )
    consistency_model.eval()

    # Initialize the WandB logger
    wandb_logger = WandbLogger(
        project="ConsistencyModelExperimentDose",  # Set your project name
        log_model=True
    )
    wandb_logger.experiment.config.update(config)

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu' if device == torch.device('cuda') else 'cpu',
        devices=1,
        max_epochs=test_config['consistency_epochs'],
        logger=wandb_logger,
        log_every_n_steps=10
    )
    # Run test
    trainer.test(consistency_model, datamodule=data_module)

    # Close WandB logging
    wandb_logger.experiment.finish()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='/home/buchwald/PycharmProjects/Diffusion_Dose/config/geometries_cond.yaml', type=str)
    args = parser.parse_args()
    test(args)
