import torch


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """

    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # Mimicking how compvis repo creates schedule
        self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
        )
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

    def add_noise_(self, original, noise, t):
        r"""
        Forward method for diffusion without inplace operations
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return: Noised image tensor
        """
        # Ensure the input tensors are not modified in-place
        original = original.clone()  # Clone the original image to avoid modifying the input tensor
        noise = noise.clone()  # Clone the noise tensor for safety

        original_shape = original.shape
        batch_size = original_shape[0]

        # Extract the corresponding sqrt_alpha_cum_prod and sqrt_one_minus_alpha_cum_prod values for the current timestep
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size).clone()
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(
            batch_size).clone()

        # Reshape them to match the dimensions of the original image
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply the noise using the forward process equation
        noised_image = sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

        return noised_image

    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the noise predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) / torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])

        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)

            return mean + sigma * z, x0

    def sample_prev_timestep_batch(self, xt, noise_pred, t):
        r"""
        Use the noise prediction by the model to get xt-1 using xt and the noise predicted
        :param xt: batch of current timestep samples (B, C, H, W)
        :param noise_pred: batch of noise predictions by the model (B, C, H, W)
        :param t: batch of timesteps (B,), where each element is the timestep for the corresponding image
        :return: xt-1 for each image in the batch
        """
        batch_size = xt.shape[0]

        # Reshape t to (B, 1, 1, 1) to match the image dimensions
        t = t.view(batch_size, 1, 1, 1)

        # Extract the corresponding sqrt_alpha_cum_prod and sqrt_one_minus_alpha_cum_prod values for each image's timestep
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t].reshape(batch_size, 1, 1, 1)
        alpha_cum_prod = self.alpha_cum_prod.to(xt.device)[t].reshape(batch_size, 1, 1, 1)
        betas_t = self.betas.to(xt.device)[t].reshape(batch_size, 1, 1, 1)
        alphas_t = self.alphas.to(xt.device)[t].reshape(batch_size, 1, 1, 1)

        # Calculate x0 (the clean image approximation)
        x0 = ((xt - (sqrt_one_minus_alpha_cum_prod * noise_pred)) / torch.sqrt(alpha_cum_prod))
        x0 = torch.clamp(x0, -1., 1.)

        # Calculate the mean for the reverse step
        mean = xt - ((betas_t * noise_pred) / sqrt_one_minus_alpha_cum_prod)
        mean = mean / torch.sqrt(alphas_t)

        # Handle the case when t == 0 (final step)
        if torch.all(t == 0):
            return mean, x0
        else:
            # Calculate variance and sigma for noise addition
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5

            # Add noise z (standard normal distribution)
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0
