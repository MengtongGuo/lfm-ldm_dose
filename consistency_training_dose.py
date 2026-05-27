import wandb
import yaml
import os
import pytorch_lightning as pl
import argparse
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from models.pl_consistency_model import LightningConsistencyModel  # Consistency model
from models.pl_ldm_dose import LightningLatentDiffusion  # The LDM module we defined earlier
from models.pl_vqvae_dose import LightningVQVAE  # VAE model
from models.unet_cond import UnetDose as Unet
from denoiser.karras_denoiser import KarrasDiffusion  # Updated Karras denoiser
from dataset.pl_dose_dataset import DoseDataModule
from utils.resample import create_named_schedule_sampler


def train(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    diffusion_model_config = config['ldm_params']
    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    consistency_config = config['consistency_params']

    # Initialize data module
    data_module = DoseDataModule(
        root_dir=dataset_config['im_path'],  # use this to point to /hdd/Josch_Data/simulations
        batch_size=train_config['ldm_batch_size'],
        num_workers=0,
        split=(0.8, 0.1, 0.1),
        downsample_factor=dataset_config['downsample_factor']
    )

    # Load the checkpoint if specified
    vqvae_checkpoint_path = train_config.get('vqvae_autoencoder_ckpt_name', None)
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
    unet_consistency.train()

    # Initialize models and scheduler
    unet_target = Unet(autoencoder_model_config['z_channels'], consistency_model_config).to(device)
    unet_target.train()

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
        loss_norm=diffusion_config["loss_norm"]
    )

    # Initialize noise scheduler
    karras_diffusion_consistency = KarrasDiffusion(
        sigma_min=consistency_config["sigma_min"],
        sigma_max=consistency_config["sigma_max"],
        rho=consistency_config["rho"],
        distillation=True,
        steps=consistency_config["num_scales"],  # 1
        loss_norm=consistency_config["loss_norm"],
        weight_schedule=consistency_config["weight_schedule"]
    )

    ldm_checkpoint_path = train_config.get('ldm_ckpt_cond_name', None)
    ldm = LightningLatentDiffusion.load_from_checkpoint(
        ldm_checkpoint_path,
        unet_model=unet_teacher,
        vae_model=vae,
        diffusion=karras_diffusion_teacher,
        learning_rate=train_config['ldm_lr'],
        plot_example_images_epoch_start=30,
        sigma_min=diffusion_config["sigma_min"],
        sigma_max=diffusion_config["sigma_max"],
        rho=diffusion_config["rho"]
    ).to(device)

    # Initialize LightningConsistencyModel
    lightning_consistency_model = LightningConsistencyModel(
        model=unet_consistency,
        diffusion=karras_diffusion_consistency,
        vae_model=vae,
        teacher_model=ldm.model,
        target_model=unet_target,
        schedule_sampler=schedule_sampler,
        teacher_diffusion=karras_diffusion_teacher,
        training_mode="consistency_distillation",
        plot_example_images_epoch_start=10,
        lr=train_config['consistency_lr'],
        num_scales=consistency_config["num_scales"],
        weight_decay=consistency_config["weight_decay"],
        ema_decay=consistency_config["ema_decay"],
        sigma_min=diffusion_config["sigma_min"],
        sigma_max=diffusion_config["sigma_max"],
        rho=diffusion_config["rho"]
    )

    lightning_consistency_model.train()

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/consistency_model/',
        filename='consistency_model-{epoch:02d}-{train/mse_loss:.4f}',
        save_last=True,
        save_top_k=0,
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val/loss',  # Metric to monitor
        patience=200,
        mode='min',
        verbose=True
    )

    # Initialize the WandB logger
    wandb_logger = WandbLogger(
        project="ConsistencyModelExperimentDose",  # Set your project name
        log_model=True
    )
    wandb_logger.experiment.config.update(config)

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu' if device == torch.device('cuda:0') else 'cpu',
        devices=[0],
        #accelerator='cpu',
        #devices=1,
        max_epochs=train_config['consistency_epochs'],
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        # gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision=32  # Enable mixed precision (16-bit floating point)
    )

    # Train model
    trainer.fit(lightning_consistency_model, data_module)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='/home/buchwald/PycharmProjects/Diffusion_Dose/config/dose_cond.yaml', type=str)
    args = parser.parse_args()
    train(args)
