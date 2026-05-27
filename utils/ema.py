class EMA:
    def __init__(self, model, decay):
        """
        Initialize the EMA (Exponential Moving Average).
        Args:
            model: The model to apply EMA to.
            decay: EMA decay rate (e.g., 0.9999).
        """
        self.decay = decay
        self.shadow = {}
        self.model = model
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Update the EMA weights based on the current model weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self):
        """
        Apply the EMA weights to the model (for evaluation or saving).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
        """
        Restore the model to its original weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])