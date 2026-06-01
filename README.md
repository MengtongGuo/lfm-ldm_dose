# Latent Diffusion and Latent Flow Matching Models for Dose Distribution Generation

This repository contains a PyTorch Lightning-based pipeline for three-dimensional dose distribution generation. It includes two latent generative approaches:

- **Latent Diffusion Model (LDM)**
- **Latent Flow Matching Model (LFM)**

Both models operate in the latent space of a pretrained **VQ-VAE** autoencoder. The generative models use CT volumes and irradiation energy values as conditional inputs to generate corresponding dose distributions.

---

## Repository Structure

```text
.
├── config/
│   └── dose_cond.yaml
├── dataset/
├── denoiser/
├── models/
├── utils/
├── diff_env.yml
├── requirements.txt
├── vqvae_dose_training.py
├── ldm_training_dose.py
├── ldm_testing_dose.py
├── fm_training_dose.py
├── fm_testing_dose.py
├── resample.py
├── sweep_runner.py
└── sweep_vqvae.yaml
```

### Main Components

- `models/`
  - Contains the model definitions for the VQ-VAE and latent generative models.
  - Includes the conditional U-Net architecture used by the LDM and LFM.

- `denoiser/`
  - Contains denoising and noise scheduling components used by the latent diffusion model.

- `dataset/`
  - Contains the data loading pipeline for CT volumes, irradiation energy values, dose volumes, and optionally precomputed latent representations.

- `utils/`
  - Contains utility functions for configuration loading and training.

- `config/dose_cond.yaml`
  - Main YAML configuration file for training and evaluation.

- `vqvae_dose_training.py`
  - Training script for the VQ-VAE autoencoder.

- `ldm_training_dose.py`
  - Training script for the latent diffusion model.

- `ldm_testing_dose.py`
  - Evaluation script for the latent diffusion model.

- `fm_training_dose.py`
  - Training script for the latent flow matching model.

- `fm_testing_dose.py`
  - Evaluation script for the latent flow matching model.

---

## Setup

### Installation with Conda

Create the environment using:

```bash
conda env create -f diff_env.yml
```

Activate the environment using:

```bash
conda activate diff_env
```

The required Python packages are also listed in:

```text
requirements.txt
```

Make sure that you have a working [Weights & Biases](https://wandb.ai/) account if you wish to use the integrated experiment logging.

---

## Data and Preprocessing

The dataset is based on the HNSCC dataset from The Cancer Imaging Archive. It includes CT images and corresponding Geant4 simulation volumes. The simulations represent beams hitting the CT anatomy from a fixed angle at different irradiation energy levels.

Each sample contains:

- a three-dimensional CT cube,
- an irradiation energy value,
- and the corresponding three-dimensional dose cube.

The preprocessed dataset contains 160,000 cubes:

- 200 cubes per patient,
- 100 patients,
- and 8 different energy levels.

The original cube dimensions are:

```text
100 × 100 × 100
```

The complete dataset is approximately 1.8 TB. Contact [Marcus Buchwald](mailto:marcus.buchwald@uni-heidelberg.de) for access.

---

## Training Pipeline

The training pipeline consists of two stages.

### Stage 1: Train the VQ-VAE

The VQ-VAE is trained first to learn a compressed latent representation of the dose distributions.

```bash
python vqvae_dose_training.py
```

The resulting VQ-VAE checkpoint is then used by both latent generative models.

### Stage 2A: Train the Latent Diffusion Model

To train the LDM, run:

```bash
python ldm_training_dose.py --config ./config/dose_cond.yaml --seed 42
```

The LDM learns to generate dose distributions through an iterative denoising process in the latent space.

### Stage 2B: Train the Latent Flow Matching Model

To train the LFM, run:

```bash
python fm_training_dose.py --config ./config/dose_cond.yaml --seed 42
```

The LFM learns a continuous flow in the latent space that transforms an initial noise distribution into the target dose distribution.

---

## Evaluation

### Evaluate the Latent Diffusion Model

```bash
python ldm_testing_dose.py --config ./config/dose_cond.yaml
```

### Evaluate the Latent Flow Matching Model

```bash
python fm_testing_dose.py --config ./config/dose_cond.yaml
```

---

## Main Arguments

- `--config`
  - Path to the YAML configuration file.
  - Default: `./config/dose_cond.yaml`

- `--seed`
  - Random seed for reproducibility.
  - Default: `42`

---

## Configuration

The training setup is controlled through:

```text
config/dose_cond.yaml
```

The configuration file contains model, dataset, and training parameters, such as:

- dataset paths,
- checkpoint paths,
- batch size,
- learning rate,
- number of training epochs,
- latent dimensions,
- and logging settings.

---

## Features

- Reproducible training through automatic random seed setting
- Three-dimensional dose distribution generation
- VQ-VAE-based latent representation learning
- Conditional generation using CT volumes and irradiation energy values
- Latent diffusion training and evaluation
- Latent flow matching training and evaluation
- Mixed-precision training
- Weights & Biases integration
- Early stopping and checkpointing
- Configurable model and training parameters

---

## Notes

- CUDA is used automatically when an available GPU is detected.
- The VQ-VAE checkpoint should be trained before training the LDM or LFM.
- Batch size, learning rate, number of epochs, and other hyperparameters can be modified in the YAML configuration file.

---

## License

This project is licensed under the MIT License.

Parts of the latent diffusion implementation are based on the [OpenAI Consistency Models repository](https://github.com/openai/consistency_models/tree/main).
