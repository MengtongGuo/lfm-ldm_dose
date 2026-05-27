import os, argparse, yaml, tempfile, subprocess, sys

PROJ = "/content/drive/MyDrive/Diffusion_Dose_Colab/Diffusion_Dose"
os.chdir(PROJ)

parser = argparse.ArgumentParser()
parser.add_argument("--base_config", type=str, default=f"{PROJ}/config/dose_cond.yaml")
parser.add_argument("--autoencoder_lr", type=float)
parser.add_argument("--autoencoder_batch_size", type=int)
parser.add_argument("--commitment_beta", type=float)
parser.add_argument("--perceptual_weight", type=float)
parser.add_argument("--autoencoder_epochs", type=int)
args, _ = parser.parse_known_args()

with open(args.base_config, "r") as f:
    cfg = yaml.safe_load(f)

tp = cfg.setdefault("train_params", {})
def set_if(k, v):
    if v is not None: tp[k] = v
set_if("autoencoder_lr", args.autoencoder_lr)
set_if("autoencoder_batch_size", args.autoencoder_batch_size)
set_if("commitment_beta", args.commitment_beta)
set_if("perceptual_weight", args.perceptual_weight)
set_if("autoencoder_epochs", args.autoencoder_epochs)

fd, tmp_path = tempfile.mkstemp(suffix=".yaml"); os.close(fd)
with open(tmp_path, "w") as f: yaml.safe_dump(cfg, f)

script = f"{PROJ}/vqvae_dose_training.py"
cmd = [sys.executable, script, "--config", tmp_path]
print("Running:", " ".join(cmd))
sys.exit(subprocess.call(cmd, cwd=PROJ))
