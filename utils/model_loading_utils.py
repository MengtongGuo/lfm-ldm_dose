import torch


def load_vqvae_checkpoint(model, checkpoint_path, device, prefix='vqvae.'):
    """
    Load a VQVAE checkpoint with optional prefix removal from state_dict keys.

    Parameters:
    - model: torch.nn.Module
        The VQVAE model instance into which to load the state_dict.
    - checkpoint_path: str
        Path to the checkpoint file.
    - device: torch.device
        The device on which to load the model (e.g., 'cuda' or 'cpu').
    - prefix: str, optional
        The prefix to remove from the state_dict keys if it exists. Default is 'vqvae.'.

    Returns:
    - model: torch.nn.Module
        The model with loaded weights.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract the state_dict from the checkpoint
    state_dict = checkpoint['state_dict']

    # Remove prefix from state_dict keys if present
    new_state_dict = {k.replace(prefix, ''): v for k, v in state_dict.items()}

    # Load the modified state_dict into the model
    model.load_state_dict(new_state_dict)
    print(f"Loaded VQVAE checkpoint from {checkpoint_path}")

    return model
