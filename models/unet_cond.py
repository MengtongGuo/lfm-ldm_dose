import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.config_utils import *
from models.blocks import get_time_embedding
import os


class ConditionalInstanceNorm(nn.Module):
    def __init__(self, num_channels, embedding_dim, dim_type='2d'):
        super().__init__()
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.dim_type = dim_type

        norm_dict = {
            '2d': nn.InstanceNorm2d,
            '3d': nn.InstanceNorm3d
        }

        self.inst_norm = norm_dict[dim_type](num_channels, affine=False)
        self.scale = nn.Linear(embedding_dim, num_channels)
        self.shift = nn.Linear(embedding_dim, num_channels)

    def forward(self, x, emb):
        gamma = self.scale(emb)
        beta = self.shift(emb)

        if self.dim_type == '2d':
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        elif self.dim_type == '3d':
            gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1, 1]
            beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return self.inst_norm(x) * (1 + gamma) + beta

class CrossAttention(nn.Module):
    def __init__(self, channels, c_emb_dim, num_heads=8, dim_type='2d'):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.dim_type = dim_type
        assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"

        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(c_emb_dim, channels)
        self.value = nn.Linear(c_emb_dim, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, c_emb):
        if self.dim_type == '2d':
            batch_size, _, height, width = x.shape
            x_flat = x.view(batch_size, self.channels, -1).permute(0, 2, 1)
        else:  # 3d
            batch_size, _, depth, height, width = x.shape
            x_flat = x.view(batch_size, self.channels, -1).permute(0, 2, 1)

        query = self.query(x_flat).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(c_emb).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(c_emb).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, value)

        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.channels)
        out = self.proj(out)

        if self.dim_type == '2d':
            out = out.permute(0, 2, 1).view(batch_size, self.channels, height, width)
        else:  # 3d
            out = out.permute(0, 2, 1).view(batch_size, self.channels, depth, height, width)

        return out

class ResidualBlockNoCond(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, dim_type='2d'):
        super().__init__()
        self.dim_type = dim_type
        self.dropout_rate = dropout

        conv_dict = {
            '2d': nn.Conv2d,
            '3d': nn.Conv3d
        }
        norm_dict = {
            '2d': nn.InstanceNorm2d,
            '3d': nn.InstanceNorm3d
        }
        dropout_dict = {
            '2d': nn.Dropout2d,
            '3d': nn.Dropout3d
        }

        self.norm1 = norm_dict[dim_type](in_channels)
        self.conv1 = conv_dict[dim_type](in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = norm_dict[dim_type](out_channels)
        self.conv2 = conv_dict[dim_type](out_channels, out_channels, kernel_size=3, padding=1)

        self.dropout = dropout_dict[dim_type](dropout) if dropout > 0 else nn.Identity()

        self.res_conv = conv_dict[dim_type](in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.dropout(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.dropout(h)

        return h + self.res_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, c_emb_dim, dropout=0.0, dim_type='2d'):
        super().__init__()
        self.dim_type = dim_type
        self.dropout_rate = dropout

        conv_dict = {
            '2d': nn.Conv2d,
            '3d': nn.Conv3d
        }
        dropout_dict = {
            '2d': nn.Dropout2d,
            '3d': nn.Dropout3d
        }
        self.dropout = dropout_dict[dim_type](dropout) if dropout > 0 else nn.Identity()

        self.norm1 = ConditionalInstanceNorm(in_channels, t_emb_dim + c_emb_dim, dim_type=dim_type)
        self.conv1 = conv_dict[dim_type](in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = ConditionalInstanceNorm(out_channels, t_emb_dim + c_emb_dim, dim_type=dim_type)
        self.conv2 = conv_dict[dim_type](out_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )
        self.cond_emb_proj = nn.Sequential(
            nn.Linear(c_emb_dim, c_emb_dim),
            nn.BatchNorm1d(c_emb_dim),
            nn.SiLU(),
            nn.Linear(c_emb_dim, c_emb_dim),
            nn.BatchNorm1d(c_emb_dim),
            nn.SiLU()
        )

        self.res_conv = conv_dict[dim_type](in_channels, out_channels,
                                            kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Use CrossAttention
        self.cross_attention1 = CrossAttention(out_channels, c_emb_dim, dim_type=dim_type)
        self.cross_attention2 = CrossAttention(out_channels, c_emb_dim, dim_type=dim_type)

    def forward(self, x, t_emb, c_emb):
        emb = torch.cat([self.time_emb_proj(t_emb), self.cond_emb_proj(c_emb)], dim=1)
        h = self.norm1(x, emb)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.dropout(h)  # Apply dropout

        # Apply CrossAttention
        h = h + self.cross_attention1(h, self.cond_emb_proj(c_emb))

        h = self.norm2(h, emb)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.dropout(h)  # Apply dropout
        h = h + self.res_conv(x)

        # Apply CrossAttention
        h = h + self.cross_attention2(h, self.cond_emb_proj(c_emb))

        return h


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, dim_type='2d'):
        super().__init__()
        norm_dict = {
            '2d': nn.GroupNorm,
            '3d': nn.GroupNorm
        }
        self.dim_type = dim_type
        self.norm = norm_dict[dim_type](32, in_channels)

        conv_dict = {
            '2d': nn.Conv2d,
            '3d': nn.Conv3d
        }

        self.qkv = conv_dict[dim_type](in_channels, in_channels * 3, kernel_size=1)
        self.proj = conv_dict[dim_type](in_channels, in_channels, kernel_size=1)
        self.num_heads = 8

    def forward(self, x):
        if self.dim_type == '2d':
            B, C, H, W = x.shape
        else:  # 3d
            B, C, D, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)

        if self.dim_type == '2d':
            qkv = rearrange(qkv, 'b (qkv c) h w -> qkv b c (h w)', qkv=3, c=C)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
            k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
            v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
            attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
            attn = attn.softmax(dim=-1)
            out = attn @ v
            out = out.reshape(B, C, H, W)
        else:  # 3d
            qkv = rearrange(qkv, 'b (qkv c) d h w -> qkv b c (d h w)', qkv=3, c=C)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q.reshape(B, self.num_heads, C // self.num_heads, D * H * W)
            k = k.reshape(B, self.num_heads, C // self.num_heads, D * H * W)
            v = v.reshape(B, self.num_heads, C // self.num_heads, D * H * W)
            attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
            attn = attn.softmax(dim=-1)
            out = attn @ v
            out = out.reshape(B, C, D, H, W)

        out = self.proj(out)
        return x + out


class ConditionHead(nn.Module):
    def __init__(self, in_channels, num_cond_variables, dim_type='2d'):
        super().__init__()
        self.in_channels = in_channels
        self.num_cond_variables = num_cond_variables
        self.dim_type = dim_type

        conv_dict = {
            '2d': nn.Conv2d,
            '3d': nn.Conv3d
        }
        norm_dict = {
            '2d': nn.BatchNorm2d,
            '3d': nn.BatchNorm3d
        }
        pool_dict = {
            '2d': nn.AdaptiveAvgPool2d,
            '3d': nn.AdaptiveAvgPool3d
        }

        self.conv1 = conv_dict[dim_type](in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_dict[dim_type](64)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(64, 128, stride=2)
        self.layer2 = self._make_layer(128, 256, stride=2)
        self.layer3 = self._make_layer(256, 512, stride=2)

        self.avg_pool = pool_dict[dim_type]((1, 1) if dim_type == '2d' else (1, 1, 1))
        self.fc = nn.Linear(512, num_cond_variables)

    def _make_layer(self, in_channels, out_channels, stride):
        conv_dict = {
            '2d': nn.Conv2d,
            '3d': nn.Conv3d
        }
        norm_dict = {
            '2d': nn.BatchNorm2d,
            '3d': nn.BatchNorm3d
        }

        layers = [
            conv_dict[self.dim_type](in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            norm_dict[self.dim_type](out_channels),
            nn.ReLU(),
            conv_dict[self.dim_type](out_channels, out_channels, kernel_size=3, padding=1),
            norm_dict[self.dim_type](out_channels)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        h = self.fc(x)

        # Apply softmax and sigmoid without in-place operations
        h_class = F.softmax(h[:, :4], dim=1)
        h_value = torch.sigmoid(h[:, 4:])
        h = torch.cat([h_class, h_value], dim=1)

        return h


class Unet(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.c_emb_dim = im_channels  # condition embedding dimension
        self.down_sample = model_config['down_sample']
        self.up_sample = list(reversed(self.down_sample))
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.dropout = model_config['dropout_rate']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        self.condition_config = get_config_value(model_config, 'condition_config', None)
        self.dim_type = model_config.get('dim_type', '2d')

        # Dynamic module selection based on dim_type
        conv_dict = {
            '2d': nn.Conv2d,
            '3d': nn.Conv3d
        }
        conv_transpose_dict = {
            '2d': nn.ConvTranspose2d,
            '3d': nn.ConvTranspose3d
        }

        self.num_cond_variables = self.condition_config['tensor_condition_config']['num_cond_variables']
        self.num_objects = self.condition_config['tensor_condition_config']['num_objects']

        # Dynamic convolution based on dim_type
        self.conv_in = conv_dict[self.dim_type](im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        "In case the condition is not appended to each image pixel, set c_emb_dim = t_emb_dim"
        self.c_emb_dim = self.t_emb_dim

        self.c_proj = nn.Sequential(
            nn.Linear(self.num_cond_variables*self.num_objects, self.c_emb_dim),
            nn.SiLU(),
            nn.Linear(self.c_emb_dim, self.c_emb_dim)
        )

        self.downs = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(self.down_channels[i], self.down_channels[i + 1],
                                  self.t_emb_dim, self.c_emb_dim, dropout=self.dropout, dim_type=self.dim_type),
                    SelfAttentionBlock(self.down_channels[i + 1], dim_type=self.dim_type) if self.attns[
                        i] else nn.Identity(),
                    conv_dict[self.dim_type](self.down_channels[i + 1], self.down_channels[i + 1],
                                             kernel_size=4, stride=2, padding=1)
                ])
            )

        # Define mid layers
        self.mids = nn.ModuleList()
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                ResidualBlock(self.mid_channels[i], self.mid_channels[i + 1],
                              self.t_emb_dim, self.c_emb_dim, dropout=self.dropout, dim_type=self.dim_type)
            )

        # Define up layers
        self.ups = nn.ModuleList()
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                nn.ModuleList([
                    conv_transpose_dict[self.dim_type](self.down_channels[i + 1], self.down_channels[i + 1],
                                                       kernel_size=4, stride=2, padding=1),
                    ResidualBlock(self.down_channels[i + 1] * 2, self.down_channels[i],
                                  self.t_emb_dim, self.c_emb_dim, dropout=self.dropout, dim_type=self.dim_type),
                    SelfAttentionBlock(self.down_channels[i], dim_type=self.dim_type) if self.attns[
                        i] else nn.Identity()
                ])
            )

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = conv_dict[self.dim_type](self.conv_out_channels, im_channels, kernel_size=3, padding=1)

        # Add a CNN-based head for condition reconstruction
        self.condition_head = ConditionHead(self.c_emb_dim, self.num_cond_variables, dim_type=self.dim_type)

    def forward(self, x, t, cond_input):
        y = cond_input['tensor'].to(x.device)
        y = self.c_proj(y)
        out = self.conv_in(x)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        downs_outs = []
        for down_block, attn, downsample in self.downs:
            out = down_block(out, t_emb, y)
            out = attn(out)
            downs_outs.append(out)
            out = downsample(out)

        for mid_block in self.mids:
            out = mid_block(out, t_emb, y)

        for upsample, up_block, attn in self.ups:
            out = upsample(out)
            out = torch.cat([out, downs_outs.pop()], dim=1)
            out = up_block(out, t_emb, y)
            out = attn(out)

        out = self.norm_out(out)
        out = F.silu(out)
        out = self.conv_out(out)

        return out

    def reconstruct_condition(self, noise_pred):
        y_recon = self.condition_head(noise_pred)
        return y_recon


class UnetDose(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.c_emb_dim = self.t_emb_dim
        self.down_sample = model_config['down_sample']
        self.up_sample = list(reversed(self.down_sample))
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.dropout = model_config['dropout_rate']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        self.dim_type = model_config.get('dim_type', '3d')
        self.image_condition_model_start_filters = model_config.get('image_condition_model_start_filters', 16)
        self.ct_encoder_layers = model_config.get('ct_encoder_layers', 5)

        conv_dict = {'2d': nn.Conv2d, '3d': nn.Conv3d}
        conv_transpose_dict = {'2d': nn.ConvTranspose2d, '3d': nn.ConvTranspose3d}

        self.conv_in = conv_dict[self.dim_type](im_channels, self.down_channels[0], kernel_size=3, padding=1)

        # Time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # Extended CT encoder for d x d x d dimensional cubes
        ct_c_in_channels = 1
        ct_encoder = []
        in_channels = ct_c_in_channels
        out_channels = self.image_condition_model_start_filters
        for i in range(self.ct_encoder_layers):
            ct_encoder.append(
                ResidualBlockNoCond(in_channels, out_channels, dropout=0.0, dim_type=self.dim_type)
            )
            ct_encoder.append(nn.ReLU())
            ct_encoder.append(conv_dict[self.dim_type](out_channels, out_channels, kernel_size=3, stride=2, padding=1))
            in_channels = out_channels
            out_channels *= 2

        self.ct_encoder = nn.Sequential(
            *ct_encoder,
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, self.c_emb_dim),
            nn.SiLU(),
            nn.Linear(self.c_emb_dim, self.c_emb_dim)
        )

        # energy nn module
        self.energy_proj = nn.Sequential(
            nn.Linear(1, self.c_emb_dim),
            nn.SiLU(),
            nn.Linear(self.c_emb_dim, self.c_emb_dim)
        )

        self.downs = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                nn.ModuleList([
                    ResidualBlock(self.down_channels[i], self.down_channels[i + 1],
                                  self.t_emb_dim, self.c_emb_dim, dropout=self.dropout, dim_type=self.dim_type),
                    SelfAttentionBlock(self.down_channels[i + 1], dim_type=self.dim_type) if self.attns[i] else nn.Identity(),
                    conv_dict[self.dim_type](self.down_channels[i + 1], self.down_channels[i + 1],
                                             kernel_size=4, stride=2, padding=1)
                ])
            )

        self.mids = nn.ModuleList()
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                ResidualBlock(self.mid_channels[i], self.mid_channels[i + 1],
                              self.t_emb_dim, self.c_emb_dim, dropout=self.dropout, dim_type=self.dim_type)
            )

        self.ups = nn.ModuleList()
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                nn.ModuleList([
                    conv_transpose_dict[self.dim_type](self.down_channels[i + 1], self.down_channels[i + 1],
                                                       kernel_size=4, stride=2, padding=1),
                    ResidualBlock(self.down_channels[i + 1] * 2, self.down_channels[i],
                                  self.t_emb_dim, self.c_emb_dim, dropout=self.dropout, dim_type=self.dim_type),
                    SelfAttentionBlock(self.down_channels[i], dim_type=self.dim_type) if self.attns[i] else nn.Identity()
                ])
            )

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = conv_dict[self.dim_type](self.conv_out_channels, im_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond_input):
        ct = cond_input['ct'].to(x.device)           # (B, 1, D, H, W)
        energy = cond_input['energy'].to(x.device)   # (B, 1)

        ct_feat = self.ct_encoder(ct)
        energy_feat = self.energy_proj(energy)
        cond = ct_feat + energy_feat  # (B, c_emb_dim)

        out = self.conv_in(x)

        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        downs_outs = []
        for down_block, attn, downsample in self.downs:
            out = down_block(out, t_emb, cond)
            out = attn(out)
            downs_outs.append(out)
            out = downsample(out)

        for mid_block in self.mids:
            out = mid_block(out, t_emb, cond)

        for upsample, up_block, attn in self.ups:
            out = upsample(out)

            # ✅ 【唯一修改点】：在cat之前做安全裁剪
            skip = downs_outs[-1]
            if out.shape[-3:] != skip.shape[-3:]:
                min_d = min(out.shape[-3], skip.shape[-3])
                min_h = min(out.shape[-2], skip.shape[-2])
                min_w = min(out.shape[-1], skip.shape[-1])
                out = out[..., :min_d, :min_h, :min_w]
                downs_outs[-1] = skip[..., :min_d, :min_h, :min_w]
            # -----------------------------------------

            out = torch.cat([out, downs_outs.pop()], dim=1)
            out = up_block(out, t_emb, cond)
            out = attn(out)

        out = self.norm_out(out)
        out = F.silu(out)
        out = self.conv_out(out)

        return out


def add_noise(x, noise, t, T):
    alpha = t / T
    return x * (1 - alpha) + noise * alpha


# Pre-training function
def pretrain_condition_head(model, vae, dataloader, optimizer, num_epochs, scheduler):
    model.train()
    criterion_cont = nn.MSELoss()
    criterion_cat = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        for x, cond_input in dataloader:
            optimizer.zero_grad()
            with torch.no_grad():
                x, _ = vae.encode(x.to('cuda:0'), cond_input['tensor'].to('cuda:0'))
            t = torch.randint(0, scheduler.num_timesteps // 2, (x.size(0),), device=x.device)
            noise = torch.randn_like(x, device=x.device)
            noised_x = scheduler.add_noise(x, noise, t)

            y_recon = model.reconstruct_condition(noised_x)

            loss_cont = criterion_cont(y_recon[:, 4:], cond_input['tensor'][:, 4:].to('cuda:0'))

            target_indices = torch.argmax(cond_input['tensor'][:, :4], dim=1).to('cuda:0')
            loss_cat = criterion_cat(y_recon[:, :4], target_indices.long())

            loss = 50 * loss_cont + loss_cat

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


# Training function
def train_full_model(model, vae, dataloader, optimizer, num_epochs, lambda_recon, scheduler, train_config):
    model.train()
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        for x, cond_input in dataloader:
            optimizer.zero_grad()
            with torch.no_grad():
                x, _ = vae.encode(x.to('cuda:0'), cond_input['tensor'].to('cuda:0'))

            t = torch.randint(0, scheduler.num_timesteps, (x.size(0),), device=x.device)

            noise = torch.randn_like(x, device=x.device)
            noised_x = scheduler.add_noise(x, noise, t)

            noise_pred = model(noised_x, t, cond_input)
            y_recon = model.reconstruct_condition(noise_pred)

            recon_weight = 1 - (t.float() / scheduler.num_timesteps)
            recon_loss = criterion(y_recon, cond_input['tensor'].to('cuda:0')) * recon_weight.mean()
            noise_loss = criterion(noise_pred, noise)

            loss = noise_loss + lambda_recon * recon_loss
            loss.backward()
            optimizer.step()

        print(f'Finished epoch:{epoch + 1}/{num_epochs} | Loss : {loss.item()}')

        torch.save(model.state_dict(), os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']))
