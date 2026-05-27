import yaml
import os
import argparse
import torch
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.unet_fm import UnetDose as UnetFM
from models.pl_vqvae_dose import LightningVQVAE  # Your VAE implementation
from dataset.pl_dose_dataset import DoseDataModule
from models.pl_fm_dose import LightningLatentFlowMatching

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


    dataset_config = config['dataset_params']
    fm_config = config['fm_params']
    unet_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    test_config = config['train_params']

    # Initialize the test data module
    data_module = DoseDataModule(
        root_dir=dataset_config['im_path'],  # use this to point to /hdd/Josch_Data/simulations
        batch_size=test_config['fm_batch_size'],
        num_workers=0,
        #split=(0.8, 0.1, 0.1),
        test_ratio=0.1, 
        num_folds=5,
        current_fold=0,
        seed=args.seed,
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

    """Load the FM checkpoint if specified"""
    # Initialize models and scheduler
    fm_unet = UnetFM(autoencoder_model_config['z_channels'], unet_config).to(device)
    fm_unet.eval()
    

    fm_checkpoint_path = test_config.get('fm_ckpt_cond_name', None)
    if fm_checkpoint_path is not None:
        fm = LightningLatentFlowMatching.load_from_checkpoint(
        fm_checkpoint_path,
        unet_model=fm_unet,
        vae_model=vae,
        learning_rate=test_config['fm_lr'],
        num_steps=fm_config['num_steps'],
        plot_example_images_epoch_start=0,
        ode_solver=fm_config.get('ode_solver', 'euler'),
        ).to(device)
    else:
        raise ValueError("No FM checkpoint was loaded!")
    
    fm.eval()

    # Initialize the WandB logger
    wandb_logger = WandbLogger(
        project="FMExperimentDose",
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
    trainer.test(fm, data_module)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='/home/mguo/projects/Diffusion_Dose/config/dose_cond1.yaml', type=str)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    test(args)
