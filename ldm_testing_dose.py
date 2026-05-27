import yaml
import os
import argparse
import torch
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.unet_cond import UnetDose as Unet
from models.pl_vqvae_dose import LightningVQVAE  # Your VAE implementation
from denoiser.karras_denoiser import KarrasDiffusion  # Updated Karras denoiser
from dataset.pl_dose_dataset import DoseDataModule
from models.pl_ldm_dose import LightningLatentDiffusion  # The LDM module we defined earlier
from utils.resample import create_named_schedule_sampler

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    pl.seed_everything(seed, workers=True)

def test(args):
    set_seed(args.seed)
    # Load configuration file
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    test_config = config['train_params']  # use train params for testing

    # Initialize the test data module
    data_module = DoseDataModule(
        root_dir=dataset_config['im_path'],  # use this to point to /hdd/Josch_Data/simulations
        batch_size=test_config['ldm_batch_size'],
        num_workers=0,
        split=(0.8, 0.1, 0.1),
        target_dim=dataset_config['target_dim']
    )

    """Load the VAE checkpoint if specified"""
    vqvae_checkpoint_path = test_config.get('vqvae_autoencoder_ckpt_name', None)
    vae = LightningVQVAE.load_from_checkpoint(
        vqvae_checkpoint_path,
        im_channels=dataset_config['im_channels'],
        model_config=autoencoder_model_config
    ).to(device)
    vae.eval()

    """Load the LDM checkpoint if specified"""
    # Initialize models and scheduler
    unet = Unet(autoencoder_model_config['z_channels'], diffusion_model_config).to(device)
    unet.eval()

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

    ldm_checkpoint_path = test_config.get('ldm_ckpt_cond_name', None)
    if ldm_checkpoint_path is not None:
        ldm = LightningLatentDiffusion.load_from_checkpoint(
            ldm_checkpoint_path,
            unet_model=unet,
            vae_model=vae,
            diffusion=karras_diffusion,
            schedule_sampler=schedule_sampler,
            learning_rate=test_config['ldm_lr'],
            num_timesteps=diffusion_config["num_timesteps"],
            plot_example_images_epoch_start=30,
            vae_down_sample=4
        ).to(device)
    else:
        raise ValueError("No model was loaded!")
    ldm.eval()

    # Initialize the WandB logger
    wandb_logger = WandbLogger(
        project="LDMExperimentDose",
        log_model=True
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu' if device == torch.device('cuda:0') else 'cpu',
        devices=[0],
        logger=wandb_logger,
        log_every_n_steps=10,
        precision=16  # Enable mixed precision (16-bit floating point)
    )

    # Perform testing
    trainer.test(ldm, data_module)



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='/home/mguo/projects/Diffusion_Dose/config/dose_cond1.yaml', type=str)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    test(args)