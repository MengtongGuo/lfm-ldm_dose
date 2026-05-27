
# Latent Diffusion Models (LDM) and Consistency Models (CM) for Dose Distribution Generation

This repository contains a PyTorch Lightning-based training pipeline for a **Latent Diffusion Model (LDM)** applied to Dose-Distribution data.  
It includes a VQ-VAE autoencoder, a U-Net-based denoiser, a custom noise scheduler (Karras diffusion), and all necessary utilities for training and logging.

## Repository Structure

- `models/`
  - `unet_cond.py`: Conditional U-Net architecture for the diffusion model.
  - `pl_vqvae_dose.py`: PyTorch Lightning module for the VQ-VAE autoencoder.
  - `pl_ldm_dose.py`: PyTorch Lightning module for training the Latent Diffusion Model (LDM).
- `denoiser/`
  - `karras_denoiser.py`: Karras diffusion noise scheduler implementation.
- `dataset/`
  - `pl_dose_dataset.py`: Data loading pipeline with optional latent loading.
- `utils/`
  - `config_utils.py`: Utility to handle nested configuration loading.
  - `resample.py`: Implements the schedule sampler for timestep selection.
- `config/`
  - `dose_cond.yaml`: Example YAML configuration file for training setup.

---

## Setup

### Installation

```bash
conda env create -f diff_env.yml
```

Make sure you also have a working [WandB](https://wandb.ai/) account if you wish to use the integrated experiment logging.

---

## Data & Preprocessing

The dataset is based on the HNSCC dataset from The Cancer Imaging Archive and includes the CT images as well as the 
Geant4 Simulations volumes of beams hitting the CT anatomy from a fixed angle but with various energies.
The dataset is preprocessed and contains 160 000 cubes (200 cubes per patient, 100 patients, 8 different energies). 
The cube dimensions are 100x100x100.

The full dataset is 1.8 TB large. Contact [Marcus Buchwald](mailto:marcus.buchwald@uni-heidelberg.de) for access.

---

## Training

To start training, simply run:

```bash
python ldm_training_dose.py --config ./config/dose_cond.yaml --seed 42
```

However, you need a 

**Arguments:**
- `--config`: Path to the YAML configuration file. Default is `./config/dose_cond.yaml`.
- `--seed`: Random seed for reproducibility. Default is `42`.

The script will:
- Load dataset and optionally precomputed latents.
- Initialize a pretrained VQ-VAE autoencoder if a checkpoint is provided.
- Set up the U-Net denoiser and Karras diffusion noise scheduler.
- Train the LDM with mixed-precision (16-bit floating point).
- Log metrics and checkpoints to Weights & Biases.

---

## Features

- **Reproducible Training** (automatic seed setting)  
- **Configurable Models and Training Parameters** (via YAML)  
- **Mixed-Precision Training** (fp16)  
- **WandB Integration** (for experiment tracking)  
- **Early Stopping and Checkpointing**  
- **Latent-Space Diffusion** (using VQ-VAE encoded representations)  
- **Flexible Conditional Inputs**

---

## Configuration

The training configuration is controlled through a YAML file (e.g., `dose_cond.yaml`), containing sections like:

- `diffusion_params`
- `dataset_params`
- `ldm_params`
- `autoencoder_params`
- `train_params`

see config file.

## Notes

- **Device selection** (`cuda` if available, else `cpu`) is automatic.
- **Model Checkpoints** are saved to `./checkpoints/ldm/`.
- The **VQ-VAE checkpoint** should be pre-trained separately before training the LDM.
- You can modify the batch size, learning rate, number of epochs, and other hyperparameters easily through the config file.

---

## License

This code is partially based on the open-ai consistency model repository.
https://github.com/openai/consistency_models/tree/main
This project is licensed under the MIT License.
