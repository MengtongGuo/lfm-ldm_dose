import yaml
import os
import pytorch_lightning as pl
import argparse
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from dataset.pl_dose_dataset import DoseDataModule
from models.pl_vqvae_dose import LightningVQVAE  # Your Lightning VQ-VAE wrapper
torch.set_float32_matmul_precision('medium')  # or 'medium'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train(args):
    # Load config
    torch.backends.cudnn.benchmark = True

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset_params']
    vqvae_model_config = config['autoencoder_params']
    train_config = config['train_params']

    data_module = DoseDataModule(
        root_dir=dataset_config['im_path'],  # use this to point to /hdd/Josch_Data/simulations
        batch_size=train_config['autoencoder_batch_size'],
        num_workers=0,
        split=(0.8, 0.1, 0.1),
        target_dim=dataset_config['target_dim'],
    )

    # Initialize VQ-VAE model
    vqvae = LightningVQVAE(
        im_channels=dataset_config['im_channels'],
        model_config=vqvae_model_config,
        learning_rate=train_config['autoencoder_lr'],
        sample_log_interval=5
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints/vqvae/',
        filename='vqvae-{epoch:02d}-{train/total_loss:.4f}',
        save_last=True,  # Save only the last model
        save_top_k=0,  # Keep only one checkpoint (the best one)
    )

    early_stopping_callback = EarlyStopping(
        monitor='val/total_loss',  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        mode='min',  # 'min' because we want to minimize the loss
        verbose=True
    )

    # Initialize the WandB logger
    wandb_logger = WandbLogger(
        project="VQVAEExperimentDose",  # Set your project name
        log_model=True  # Logs the model checkpoints if enabled
    )
    wandb_logger.experiment.config.update(config)  # Log the config to WandB

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[3],
        max_epochs=train_config['autoencoder_epochs'],
        callbacks=[checkpoint_callback, early_stopping_callback],  # Add EarlyStopping to callbacks
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision=32,
        #precision=16,  # Enable mixed precision (16-bit floating point),
        accumulate_grad_batches=4  # Accumulate gradients over 4 batches
    )

    # Train model
    trainer.fit(vqvae, data_module)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='/home/buchwald/PycharmProjects/Diffusion_Dose/config/dose_cond.yaml', type=str)
    args = parser.parse_args()
    train(args)

