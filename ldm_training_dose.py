import yaml
import os
import pytorch_lightning as pl
import argparse
import torch
import numpy as np
import random
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from models.unet_cond import UnetDose as Unet
from models.pl_vqvae_dose import LightningVQVAE  # Your VAE implementation
from denoiser.karras_denoiser import KarrasDiffusion  # Updated Karras denoiser
from dataset.pl_dose_dataset import DoseDataModule
from models.pl_ldm_dose import LightningLatentDiffusion  # The LDM module we defined earlier
from utils.resample import create_named_schedule_sampler
import wandb


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # These settings ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    pl.seed_everything(seed, workers=True)  # Also sets seed for dataloader workers


def train(args):
    # Set random seed first thing
    set_seed(args.seed)

    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    for fold in range(5):
        print(f"\n========== Running fold {fold + 1}/5 ==========")
        set_seed(args.seed)
        wandb.finish()

        data_module = DoseDataModule(
            root_dir=dataset_config['im_path'],  # use this to point to /hdd/Josch_Data/simulations
            batch_size=train_config['ldm_batch_size'],
            num_workers=0,
            test_ratio=0.1,
            num_folds=5,
            current_fold=fold,
            seed=args.seed,
            target_dim=dataset_config['target_dim']
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

        # Initialize models and scheduler
        unet = Unet(autoencoder_model_config['z_channels'], diffusion_model_config).to(device)
        unet.train()

        schedule_sampler = create_named_schedule_sampler('uniform', diffusion_config["num_timesteps"])

        # Initialize noise scheduler
        karras_diffusion = KarrasDiffusion(
            sigma_min=diffusion_config["sigma_min"],
            sigma_max=diffusion_config["sigma_max"],
            rho=diffusion_config["rho"],
            distillation=False,
            steps=diffusion_config["num_timesteps"],
            loss_norm=diffusion_config["loss_norm"]
        )

        # Initialize LDM module
        ldm = LightningLatentDiffusion(
            unet_model=unet,
            vae_model=vae,
            diffusion=karras_diffusion,
            schedule_sampler=schedule_sampler,
            learning_rate=train_config['ldm_lr'],
            num_timesteps=diffusion_config["num_timesteps"],
            plot_example_images_epoch_start=0,
            sigma_min=diffusion_config["sigma_min"],
            sigma_max=diffusion_config["sigma_max"],
            rho=diffusion_config["rho"]
        )

        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'./checkpoints/ldm/fold_{fold}/',
            save_last=True,  # Save only the last model
            save_top_k=1,  # Keep only one checkpoint (the latest one)
            mode='min',
            monitor='val/loss'
        )

        early_stopping_callback = EarlyStopping(
            monitor='val/loss',  # Metric to monitor
            patience=50,  # Number of epochs with no improvement after which training will be stopped
            mode='min',  # 'min' because we want to minimize the loss
            verbose=True
        )

        # Initialize the WandB logger
        wandb_logger = WandbLogger(
            project="LDMExperimentDose",  # Set your project name
            name=f"20000_fold_{fold}",
            log_model=True  # Logs the model checkpoints if enabled
        )

        wandb_logger.experiment.config.update(config)  # Log the config to WandB
        wandb_logger.experiment.config.update({'random_seed': args.seed})  # Log the seed

        # Initialize trainer
        trainer = pl.Trainer(
            accelerator='gpu' if device == torch.device('cuda:0') else 'cpu',
            #accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=[0],
            max_epochs=train_config['ldm_epochs'],
            callbacks=[checkpoint_callback, early_stopping_callback],  # Add EarlyStopping to callbacks
            logger=wandb_logger,
            # gradient_clip_val=1.0,
            log_every_n_steps=10,
            precision=32  # Enable mixed precision (16-bit floating point)
        )

        # Train model
        trainer.fit(ldm, data_module)
        wandb_logger.experiment.finish()



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='/home/mguo/projects/Diffusion_Dose/config/dose_cond1.yaml',
                        type=str)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument(
        '--resume_ckpt',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    args = parser.parse_args()
    train(args)
