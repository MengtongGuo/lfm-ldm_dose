import torch
import torch.nn as nn
import torch.nn.functional as F


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class ConditionalInstanceNorm(nn.Module):
    def __init__(self, num_features, cond_dim, dims='2d'):
        super().__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim
        self.dims = dims

        # Select appropriate normalization based on dimensions
        if dims == '2d':
            self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        elif dims == '3d':
            self.instance_norm = nn.InstanceNorm3d(num_features, affine=False)
        else:
            raise ValueError("dims must be either '2d' or '3d'")

        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)

    def forward(self, x, cond):
        normalized = self.instance_norm(x)
        gamma = self.gamma(cond)
        beta = self.beta(cond)

        # Reshape based on dimensions
        if self.dims == '2d':
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
        else:  # 3d
            gamma = gamma.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            beta = beta.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        return gamma * normalized + beta


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels,
                 use_condition=False, cond_dim=8, cross_attn=False,
                 context_dim=None, dims='2d'):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.use_condition = use_condition
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim
        self.dims = dims

        # Helper function to get appropriate conv layer
        def get_conv(in_ch, out_ch, kernel_size, stride=1, padding=0):
            if dims == '2d':
                return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
            else:  # 3d
                return nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding)

        # Helper function to get appropriate norm layer
        def get_group_norm(num_channels):
            if dims == '2d':
                return nn.GroupNorm(norm_channels, num_channels)
            else:  # 3d
                return nn.GroupNorm(norm_channels, num_channels)

        if self.use_condition:
            self.resnet_conv_first = nn.ModuleList(
                [
                    nn.Sequential(
                        ConditionalInstanceNorm(in_channels if i == 0 else out_channels, cond_dim, dims),
                        nn.SiLU(),
                        get_group_norm(in_channels if i == 0 else out_channels),
                        nn.SiLU(),
                        get_conv(in_channels if i == 0 else out_channels, out_channels, 3, 1, 1),
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.resnet_conv_first = nn.ModuleList(
                [
                    nn.Sequential(
                        get_group_norm(in_channels if i == 0 else out_channels),
                        nn.SiLU(),
                        get_conv(in_channels if i == 0 else out_channels, out_channels, 3, 1, 1),
                    )
                    for i in range(num_layers)
                ]
            )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])

        if self.use_condition:
            self.resnet_conv_second = nn.ModuleList(
                [
                    nn.Sequential(
                        ConditionalInstanceNorm(out_channels, cond_dim, dims),
                        nn.SiLU(),
                        get_group_norm(out_channels),
                        nn.SiLU(),
                        get_conv(out_channels, out_channels, 3, 1, 1),
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.resnet_conv_second = nn.ModuleList(
                [
                    nn.Sequential(
                        get_group_norm(out_channels),
                        nn.SiLU(),
                        get_conv(out_channels, out_channels, 3, 1, 1),
                    )
                    for _ in range(num_layers)
                ]
            )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [get_group_norm(out_channels)
                 for _ in range(num_layers)]
            )

            self.attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [get_group_norm(out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        if self.use_condition:
            self.residual_input_conv = nn.ModuleList(
                [
                    nn.Sequential(
                        ConditionalInstanceNorm(in_channels if i == 0 else out_channels, cond_dim, dims),
                        nn.SiLU(),
                        get_conv(in_channels if i == 0 else out_channels, out_channels, 1)
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.residual_input_conv = nn.ModuleList(
                [get_conv(in_channels if i == 0 else out_channels, out_channels, 1) for i in range(num_layers)]
            )

        # Downsample conv
        if self.down_sample:
            if dims == '2d':
                self.down_sample_conv = get_conv(out_channels, out_channels, 4, 2, 1)
            else:  # 3d
                self.down_sample_conv = get_conv(out_channels, out_channels, 4, 2, 1)
        else:
            self.down_sample_conv = nn.Identity()

    def forward(self, x, t_emb=None, context=None):
        out = x

        for i in range(self.num_layers):
            resnet_input = out

            if self.use_condition:
                out = self.resnet_conv_first[i][0](out, context)
                for layer in self.resnet_conv_first[i][1:]:
                    out = layer(out)
            else:
                out = self.resnet_conv_first[i](out)

            if self.t_emb_dim is not None:
                emb_out = self.t_emb_layers[i](t_emb)
                if self.dims == '2d':
                    emb_out = emb_out[:, :, None, None]
                else:  # 3d
                    emb_out = emb_out[:, :, None, None, None]
                out = out + emb_out

            if self.use_condition:
                out = self.resnet_conv_second[i][0](out, context)
                for layer in self.resnet_conv_second[i][1:]:
                    out = layer(out)
            else:
                out = self.resnet_conv_second[i](out)

            if self.use_condition:
                out_res = self.residual_input_conv[i][0](resnet_input, context)
                for layer in self.residual_input_conv[i][1:]:
                    out_res = layer(out_res)
            else:
                out_res = self.residual_input_conv[i](resnet_input)

            out = out + out_res

            if self.attn:
                if self.dims == '2d':
                    batch_size, channels, h, w = out.shape
                    in_attn = out.reshape(batch_size, channels, h * w)
                else:  # 3d
                    batch_size, channels, d, h, w = out.shape
                    in_attn = out.reshape(batch_size, channels, d * h * w)

                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2)

                if self.dims == '2d':
                    out_attn = out_attn.reshape(batch_size, channels, h, w)
                else:  # 3d
                    out_attn = out_attn.reshape(batch_size, channels, d, h, w)

                out = out + out_attn

            if self.cross_attn:
                # Cross attention handling remains similar, just need to adapt reshape operations
                assert context is not None, "context cannot be None if cross attention layers are used"

                if self.dims == '2d':
                    batch_size, channels, h, w = out.shape
                    in_attn = out.reshape(batch_size, channels, h * w)
                else:  # 3d
                    batch_size, channels, d, h, w = out.shape
                    in_attn = out.reshape(batch_size, channels, d * h * w)

                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)

                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                context_proj = context_proj.unsqueeze(0)

                sequence_length = in_attn.shape[1]
                context_proj = context_proj.expand(sequence_length, batch_size, -1)

                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2)

                if self.dims == '2d':
                    out_attn = out_attn.reshape(batch_size, channels, h, w)
                else:  # 3d
                    out_attn = out_attn.reshape(batch_size, channels, d, h, w)

                out = out + out_attn

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels,
                 use_condition=False, cond_dim=8, cross_attn=None, context_dim=None, dims='2d'):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.use_condition = use_condition
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.dims = dims

        # Helper function to get appropriate conv layer
        def get_conv(in_ch, out_ch, kernel_size, stride=1, padding=0):
            if dims == '2d':
                return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
            else:  # 3d
                return nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding)

        # Helper function to get appropriate norm layer
        def get_group_norm(num_channels):
            if dims == '2d':
                return nn.GroupNorm(norm_channels, num_channels)
            else:  # 3d
                return nn.GroupNorm(norm_channels, num_channels)

        if self.use_condition:
            self.resnet_conv_first = nn.ModuleList(
                [
                    nn.Sequential(
                        ConditionalInstanceNorm(in_channels if i == 0 else out_channels, cond_dim, dims),
                        nn.SiLU(),
                        get_group_norm(in_channels if i == 0 else out_channels),
                        nn.SiLU(),
                        get_conv(in_channels if i == 0 else out_channels, out_channels, 3, 1, 1),
                    )
                    for i in range(num_layers + 1)
                ]
            )
        else:
            self.resnet_conv_first = nn.ModuleList(
                [
                    nn.Sequential(
                        get_group_norm(in_channels if i == 0 else out_channels),
                        nn.SiLU(),
                        get_conv(in_channels if i == 0 else out_channels, out_channels, 3, 1, 1),
                    )
                    for i in range(num_layers + 1)
                ]
            )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])

        if self.use_condition:
            self.resnet_conv_second = nn.ModuleList(
                [
                    nn.Sequential(
                        ConditionalInstanceNorm(out_channels, cond_dim, dims),
                        nn.SiLU(),
                        get_group_norm(out_channels),
                        nn.SiLU(),
                        get_conv(out_channels, out_channels, 3, 1, 1),
                    )
                    for _ in range(num_layers + 1)
                ]
            )
        else:
            self.resnet_conv_second = nn.ModuleList(
                [
                    nn.Sequential(
                        get_group_norm(out_channels),
                        nn.SiLU(),
                        get_conv(out_channels, out_channels, 3, 1, 1),
                    )
                    for _ in range(num_layers + 1)
                ]
            )

        self.attention_norms = nn.ModuleList(
            [get_group_norm(out_channels)
             for _ in range(num_layers)]
        )

        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [get_group_norm(out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        if self.use_condition:
            self.residual_input_conv = nn.ModuleList(
                [
                    nn.Sequential(
                        ConditionalInstanceNorm(in_channels if i == 0 else out_channels, cond_dim, dims),
                        nn.SiLU(),
                        get_conv(in_channels if i == 0 else out_channels, out_channels, 1)
                    )
                    for i in range(num_layers + 1)
                ]
            )
        else:
            self.residual_input_conv = nn.ModuleList(
                [get_conv(in_channels if i == 0 else out_channels, out_channels, 1) for i in range(num_layers + 1)]
            )

    def forward(self, x, t_emb=None, context=None):
        out = x

        # First resnet block
        resnet_input = out

        if self.use_condition:
            out = self.resnet_conv_first[0][0](out, context)
            for layer in self.resnet_conv_first[0][1:]:
                out = layer(out)
        else:
            out = self.resnet_conv_first[0](out)

        if self.t_emb_dim is not None:
            emb_out = self.t_emb_layers[0](t_emb)
            if self.dims == '2d':
                emb_out = emb_out[:, :, None, None]
            else:  # 3d
                emb_out = emb_out[:, :, None, None, None]
            out = out + emb_out

        if self.use_condition:
            out = self.resnet_conv_second[0][0](out, context)
            for layer in self.resnet_conv_second[0][1:]:
                out = layer(out)
        else:
            out = self.resnet_conv_second[0](out)

        if self.use_condition:
            out_res = self.residual_input_conv[0][0](resnet_input, context)
            for layer in self.residual_input_conv[0][1:]:
                out_res = layer(out_res)
        else:
            out_res = self.residual_input_conv[0](resnet_input)

        out = out + out_res

        for i in range(self.num_layers):
            # Attention Block
            if self.dims == '2d':
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
            else:  # 3d
                batch_size, channels, d, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, d * h * w)

            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2)

            if self.dims == '2d':
                out_attn = out_attn.reshape(batch_size, channels, h, w)
            else:  # 3d
                out_attn = out_attn.reshape(batch_size, channels, d, h, w)

            out = out + out_attn

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"

                if self.dims == '2d':
                    batch_size, channels, h, w = out.shape
                    in_attn = out.reshape(batch_size, channels, h * w)
                else:  # 3d
                    batch_size, channels, d, h, w = out.shape
                    in_attn = out.reshape(batch_size, channels, d * h * w)

                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                context_proj = self.context_proj[i](context)
                context_proj = context_proj.unsqueeze(0)
                sequence_length = in_attn.shape[1]
                context_proj = context_proj.expand(sequence_length, batch_size, -1)

                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2)

                if self.dims == '2d':
                    out_attn = out_attn.reshape(batch_size, channels, h, w)
                else:  # 3d
                    out_attn = out_attn.reshape(batch_size, channels, d, h, w)

                out = out + out_attn

            # Resnet Block
            resnet_input = out

            if self.use_condition:
                out = self.resnet_conv_first[i + 1][0](out, context)
                for layer in self.resnet_conv_first[i + 1][1:]:
                    out = layer(out)
            else:
                out = self.resnet_conv_first[i + 1](out)

            if self.t_emb_dim is not None:
                emb_out = self.t_emb_layers[i + 1](t_emb)
                if self.dims == '2d':
                    emb_out = emb_out[:, :, None, None]
                else:  # 3d
                    emb_out = emb_out[:, :, None, None, None]
                out = out + emb_out

            if self.use_condition:
                out = self.resnet_conv_second[i + 1][0](out, context)
                for layer in self.resnet_conv_second[i + 1][1:]:
                    out = layer(out)
            else:
                out = self.resnet_conv_second[i + 1](out)

            if self.use_condition:
                out_res = self.residual_input_conv[i + 1][0](resnet_input, context)
                for layer in self.residual_input_conv[i + 1][1:]:
                    out_res = layer(out_res)
            else:
                out_res = self.residual_input_conv[i + 1](resnet_input)

            out = out + out_res

        return out


class UpBlock(nn.Module):
    r"""
    Up conv block with attention supporting 2D and 3D operations.
    Sequence of following blocks:
    1. Upsample
    2. Concatenate Down block output
    3. Resnet block with time embedding
    4. Attention Block
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads, num_layers, attn, norm_channels,
                 use_condition=False, cond_dim=8, dims='2d'):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.use_condition = use_condition
        self.dims = dims

        # Helper functions for dynamic layer selection
        def get_conv(in_ch, out_ch, kernel_size, stride=1, padding=1):
            return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding) if dims == '2d' else \
                nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding)

        def get_conv_transpose(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1) if dims == '2d' else \
                nn.ConvTranspose3d(in_ch, out_ch, 4, 2, 1)

        def get_norm(num_channels):
            return nn.GroupNorm(norm_channels, num_channels)

        # Conditional and non-conditional first convolution layers
        if self.use_condition:
            self.resnet_conv_first = nn.ModuleList([
                nn.Sequential(
                    ConditionalInstanceNorm(in_channels if i == 0 else out_channels, cond_dim, dims),
                    nn.SiLU(),
                    get_norm(in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    get_conv(in_channels if i == 0 else out_channels, out_channels, 3),
                ) for i in range(num_layers)
            ])
        else:
            self.resnet_conv_first = nn.ModuleList([
                nn.Sequential(
                    get_norm(in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    get_conv(in_channels if i == 0 else out_channels, out_channels, 3),
                ) for i in range(num_layers)
            ])

        # Time embedding layers
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ) for _ in range(num_layers)
            ])

        # Conditional and non-conditional second convolution layers
        if self.use_condition:
            self.resnet_conv_second = nn.ModuleList([
                nn.Sequential(
                    ConditionalInstanceNorm(out_channels, cond_dim, dims),
                    nn.SiLU(),
                    get_norm(out_channels),
                    nn.SiLU(),
                    get_conv(out_channels, out_channels, 3),
                ) for _ in range(num_layers)
            ])
        else:
            self.resnet_conv_second = nn.ModuleList([
                nn.Sequential(
                    get_norm(out_channels),
                    nn.SiLU(),
                    get_conv(out_channels, out_channels, 3),
                ) for _ in range(num_layers)
            ])

        # Attention mechanism
        if self.attn:
            self.attention_norms = nn.ModuleList([
                get_norm(out_channels) for _ in range(num_layers)
            ])

            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])

        # Conditional and non-conditional residual input convolutions
        if self.use_condition:
            self.residual_input_conv = nn.ModuleList([
                nn.Sequential(
                    ConditionalInstanceNorm(in_channels if i == 0 else out_channels, cond_dim, dims),
                    nn.SiLU(),
                    get_conv(in_channels if i == 0 else out_channels, out_channels, 1),
                ) for i in range(num_layers)
            ])
        else:
            self.residual_input_conv = nn.ModuleList([
                get_conv(in_channels if i == 0 else out_channels, out_channels, 1)
                for i in range(num_layers)
            ])

        # Upsampling
        self.up_sample_conv = get_conv_transpose(in_channels, in_channels) if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None, context=None):
        # Upsample
        x = self.up_sample_conv(x)

        # Concat with Downblock output
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out

            if self.use_condition:
                out = self.resnet_conv_first[i][0](out, context)
                for layer in self.resnet_conv_first[i][1:]:
                    out = layer(out)
            else:
                out = self.resnet_conv_first[i](out)

            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None] if self.dims == '2d' else \
                    out + self.t_emb_layers[i](t_emb)[:, :, None, None, None]

            if self.use_condition:
                out = self.resnet_conv_second[i][0](out, context)
                for layer in self.resnet_conv_second[i][1:]:
                    out = layer(out)
            else:
                out = self.resnet_conv_second[i](out)

            if self.use_condition:
                out_res = self.residual_input_conv[i][0](resnet_input, context)
                for layer in self.residual_input_conv[i][1:]:
                    out_res = layer(out_res)
            else:
                out_res = self.residual_input_conv[i](resnet_input)

            # Ensure out_res has the same shape as out
            if out_res.shape != out.shape:
                out_res = F.interpolate(out_res, size=out.shape[2:], mode='nearest')

            out = out + out_res

            # Self Attention
            if self.attn:
                batch_size, channels, *spatial_dims = out.shape
                in_attn = out.reshape(batch_size, channels, -1)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, *spatial_dims)
                out = out + out_attn

        return out
