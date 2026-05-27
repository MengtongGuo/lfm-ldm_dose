import torch
import torch.nn as nn
from models.blocks import DownBlock, MidBlock, UpBlock
import torch.nn.functional as F


class VQVAE(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.use_condition = model_config['use_condition']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['attn_down']

        # Latent Dimension
        self.z_channels = model_config['z_channels']
        self.codebook_size = model_config['codebook_size']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']

        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))

        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 use_condition=self.use_condition,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              use_condition=self.use_condition,
                                              norm_channels=self.norm_channels))

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)

        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)

        # Codebook
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)
        ####################################################

        ##################### Decoder ######################

        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              use_condition=self.use_condition,
                                              norm_channels=self.norm_channels))

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               use_condition=self.use_condition,
                                               attn=self.attns[i - 1],
                                               norm_channels=self.norm_channels))

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)

    def quantize(self, x):
        B, C, H, W = x.shape

        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)

        # B, H, W, C -> B, H*W, C
        x = x.reshape(x.size(0), -1, x.size(-1))

        # Find nearest embedding/codebook vector
        # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        # (B, H*W)
        min_encoding_indices = torch.argmin(dist, dim=-1)

        # Replace encoder output with nearest codebook
        # quant_out -> B*H*W, C
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))

        # x -> B*H*W, C
        x = x.reshape((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commmitment_loss
        }
        # Straight through estimation
        quant_out = x + (quant_out - x).detach()

        # quant_out -> B, C, H, W
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        min_encoding_indices = min_encoding_indices.reshape((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices

    def encode(self, x, context=None):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out, context=context)
        for mid in self.encoder_mids:
            out = mid(out, context=context)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        out, quant_losses, _ = self.quantize(out)
        return out, quant_losses

    def decode(self, z, context=None):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out, context=context)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out, context=context)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x, context=None):
        z, quant_losses = self.encode(x, context=context)
        out = self.decode(z, context=context)
        return out, z, quant_losses


class VQVAE3D(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.dim_type = model_config.get('dim_type', '3d')
        conv_dict = {'2d': nn.Conv2d, '3d': nn.Conv3d}
        conv_transpose_dict = {'2d': nn.ConvTranspose2d, '3d': nn.ConvTranspose3d}

        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.use_condition = model_config['use_condition']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.z_channels = model_config['z_channels']
        self.codebook_size = model_config['codebook_size']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']

        self.up_sample = list(reversed(self.down_sample))

        self.encoder_conv_in = conv_dict[self.dim_type](im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.encoder_layers = nn.ModuleList([
            DownBlock(self.down_channels[i], self.down_channels[i + 1], None,
                      down_sample=self.down_sample[i], num_heads=self.num_heads,
                      num_layers=self.num_down_layers, use_condition=self.use_condition,
                      attn=self.attns[i], norm_channels=self.norm_channels, dims=self.dim_type)
            for i in range(len(self.down_channels) - 1)
        ])
        self.encoder_mids = nn.ModuleList([
            MidBlock(self.mid_channels[i], self.mid_channels[i + 1], None,
                     num_heads=self.num_heads, num_layers=self.num_mid_layers,
                     use_condition=self.use_condition, norm_channels=self.norm_channels, dims=self.dim_type)
            for i in range(len(self.mid_channels) - 1)
        ])
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = conv_dict[self.dim_type](self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)

        self.pre_quant_conv = conv_dict[self.dim_type](self.z_channels, self.z_channels, kernel_size=1)
        self.embedding = nn.Embedding(self.codebook_size, self.z_channels)

        self.post_quant_conv = conv_dict[self.dim_type](self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = conv_dict[self.dim_type](self.z_channels, self.mid_channels[-1], kernel_size=3, padding=1)
        self.decoder_mids = nn.ModuleList([
            MidBlock(self.mid_channels[i], self.mid_channels[i - 1], None,
                     num_heads=self.num_heads, num_layers=self.num_mid_layers,
                     use_condition=self.use_condition, norm_channels=self.norm_channels, dims=self.dim_type)
            for i in reversed(range(1, len(self.mid_channels)))
        ])
        self.decoder_layers = nn.ModuleList([
            UpBlock(self.down_channels[i], self.down_channels[i - 1], None,
                    up_sample=self.down_sample[i - 1], num_heads=self.num_heads,
                    num_layers=self.num_up_layers, use_condition=self.use_condition,
                    attn=self.attns[i - 1], norm_channels=self.norm_channels, dims=self.dim_type)
            for i in reversed(range(1, len(self.down_channels)))
        ])
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = conv_dict[self.dim_type](self.down_channels[0], im_channels, kernel_size=3, padding=1)

    def quantize(self, x):
        B, C, *spatial = x.shape
        x_flat = x.permute(0, *range(2, 2 + len(spatial)), 1).reshape(B, -1, C)
        dist = torch.cdist(x_flat, self.embedding.weight[None].expand(B, -1, -1))
        indices = torch.argmin(dist, dim=-1)
        quant = self.embedding(indices.view(-1)).view(B, *spatial, C).permute(0, -1, *range(1, len(spatial)+1))
        x_orig = x.reshape(B, C, -1).permute(0, 2, 1)
        q_flat = quant.reshape(B, C, -1).permute(0, 2, 1)
        commit_loss = (q_flat.detach() - x_orig).pow(2).mean()
        codebook_loss = (q_flat - x_orig.detach()).pow(2).mean()
        quant = x + (quant - x).detach()
        return quant, {'codebook_loss': codebook_loss, 'commitment_loss': commit_loss}, indices

    def encode(self, x, context=None):
        x = self.encoder_conv_in(x)
        for layer in self.encoder_layers:
            x = layer(x, context=context)
        for mid in self.encoder_mids:
            x = mid(x, context=context)
        x = self.encoder_norm_out(x)
        x = nn.SiLU()(x)
        x = self.encoder_conv_out(x)
        x = self.pre_quant_conv(x)
        z, losses, _ = self.quantize(x)
        return z, losses

    def decode(self, z, context=None, image_shape=None):
        x = self.post_quant_conv(z)
        x = self.decoder_conv_in(x)
        for mid in self.decoder_mids:
            x = mid(x, context=context)
        for up in self.decoder_layers:
            x = up(x, context=context)
        x = self.decoder_norm_out(x)
        x = nn.SiLU()(x)
        x = self.decoder_conv_out(x)

        if image_shape is not None:
            # Crop or pad to match the target shape
            current_shape = x.shape[-3:]
            diff = [target - current for target, current in zip(image_shape[-3:], current_shape)]

            pad = []
            for d in reversed(diff):
                if d > 0:
                    pad.extend([d // 2, d - d // 2])
                else:
                    pad.extend([0, 0])
            if any(pad):
                x = F.pad(x, pad, mode="reflect")

            # Crop if needed
            crop = [(-d if d < 0 else 0) for d in diff]
            if any(crop):
                d, h, w = x.shape[-3:]
                crop_d, crop_h, crop_w = crop
                x = x[..., crop_d//2:d - (crop_d - crop_d//2),
                         crop_h//2:h - (crop_h - crop_h//2),
                         crop_w//2:w - (crop_w - crop_w//2)]

        return x

    def forward(self, x, context=None):
        # Save original size
        original_shape = x.shape[-3:]

        # Compute padding for divisibility by 2^n where n is total downsampling steps
        down_factor = 2 ** sum(self.down_sample)
        padded_shape = [((d + down_factor - 1) // down_factor) * down_factor for d in original_shape]
        pad_dims = [p - o for p, o in zip(padded_shape, original_shape)]

        # Apply symmetric padding
        pad = []
        for p in reversed(pad_dims):
            pad.extend([p // 2, p - p // 2])  # [left, right] for each dim
        x_padded = F.pad(x, pad, mode="reflect")

        # Encode and decode
        z, quant_losses = self.encode(x_padded, context=context)
        recon_padded = self.decode(z, context=context)

        # Crop to original shape
        d, h, w = recon_padded.shape[-3:]
        crop_d = (d - original_shape[0]) // 2
        crop_h = (h - original_shape[1]) // 2
        crop_w = (w - original_shape[2]) // 2

        recon = recon_padded[..., crop_d:crop_d + original_shape[0],
                crop_h:crop_h + original_shape[1],
                crop_w:crop_w + original_shape[2]]

        return recon, z, quant_losses
