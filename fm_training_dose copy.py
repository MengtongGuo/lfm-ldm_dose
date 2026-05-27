import yaml
import os
import pytorch_lightning as pl
import argparse
import torch
import numpy as np
import random
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from models.unet_fm import UnetDose as UnetFM
from models.pl_vqvae_dose import LightningVQVAE  # Your VAE implementation
from dataset.pl_dose_dataset import DoseDataModule
from models.pl_fm_dose import LightningLatentFlowMatching


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


    dataset_config = config['dataset_params']
    fm_config = config['fm_params']
    unet_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    data_module = DoseDataModule(
        root_dir=dataset_config['im_path'],  # use this to point to /hdd/Josch_Data/simulations
        batch_size=train_config['fm_batch_size'],
        num_workers=0,
        split=(0.8, 0.1, 0.1),
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

    # Initialize models
    fm_unet = UnetFM(autoencoder_model_config['z_channels'], unet_config).to(device)
    fm_unet.train()

    #schedule_sampler = create_named_schedule_sampler('uniform', diffusion_config["num_timesteps"])

    ## Initialize noise scheduler

    # Initialize LDM module
    fm = LightningLatentFlowMatching(
        unet_model=fm_unet,
        vae_model=vae,
        learning_rate=train_config['fm_lr'],
        num_steps=fm_config['num_steps'],
        plot_example_images_epoch_start=0,
        ode_solver=fm_config.get('ode_solver', 'euler'),
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/fm/',
        filename='fm-{epoch:02d}-{train/loss:.4f}',
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
        project="FMExperimentDose",  # Set your project name
        log_model=True  # Logs the model checkpoints if enabled
    )

    wandb_logger.experiment.config.update(config)  # Log the config to WandB
    wandb_logger.experiment.config.update({'random_seed': args.seed})  # Log the seed

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu' if device == torch.device('cuda:0') else 'cpu',
        #accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[0],
        max_epochs=train_config['fm_epochs'],
        callbacks=[checkpoint_callback, early_stopping_callback],  # Add EarlyStopping to callbacks
        logger=wandb_logger,
        # gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision=32  # Enable mixed precision (16-bit floating point)
    )

    # Train model
    trainer.fit(fm, data_module)
    #trainer.test(fm, data_module)



if __name__ == '__main__':
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Arguments for Flow Matching training')
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
