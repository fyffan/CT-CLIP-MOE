from pathlib import Path
import copy
import math
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.autograd import grad as torch_grad
from torchvision import transforms as T, utils


import torchvision

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import VectorQuantize

from transformer_maskgit.attention import Attention, Transformer, ContinuousPositionBias

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

def cast_tuple(val, l = 1):
    return val if isinstance(val, tuple) else (val,) * l

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    device=torch.device('cuda')
    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def l2norm(t):
    return F.normalize(t, dim = -1)

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# ctvit - 3d ViT with factorized spatial and temporal attention made into an vqgan-vae autoencoder

def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    device=torch.device('cuda')
    batch_indices = torch.arange(batch, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

# CTViT - Continuous Tokenized Video Transformer
# 假设dim为patch特征维度，num_experts为专家数，token_num为learnable token数

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=4, heads=8, mlp_ratio=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*mlp_ratio, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
    def forward(self, x):
        # x: [B, N, D]
        return self.encoder(x)

class MoE(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        # x: [B, N, D]
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [B, N, num_experts]
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-2)  # [B, N, num_experts, D]
        out = (gate_scores.unsqueeze(-1) * expert_outs).sum(dim=-2)  # [B, N, D]
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, dim, token_num, depth=4, heads=8, mlp_ratio=4):
        super().__init__()
        self.learnable_tokens = nn.Parameter(torch.randn(1, token_num, dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*mlp_ratio, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

    def forward(self, memory):
        # memory: [B, N, D]
        B = memory.size(0)
        tokens = self.learnable_tokens.expand(B, -1, -1)  # [B, token_num, D]
        out = self.decoder(tgt=tokens, memory=memory)
        return out

class TaskSpecificExperts(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_experts)])
    def forward(self, x):
        # x: [B, N, D]
        return [expert(x) for expert in self.experts]  # List of [B, N, D]

class DynamicRouting(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.routing = nn.Linear(dim, num_experts)
    def forward(self, x):
        # x: [B, N, D]
        weights = F.softmax(self.routing(x), dim=-1)  # [B, N, num_experts]
        return weights

class Fusion(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.fusion = nn.Linear(dim * (num_experts + 1), dim)
    def forward(self, expert_outputs, routing_weights, shared_expert):
        # expert_outputs: List of [B, N, D]
        # routing_weights: [B, N, num_experts]
        # shared_expert: [B, N, D]
        expert_stack = torch.stack(expert_outputs, dim=-2)  # [B, N, num_experts, D]
        weighted = (routing_weights.unsqueeze(-1) * expert_stack).sum(dim=-2)  # [B, N, D]
        concat = torch.cat([weighted, shared_expert], dim=-1)  # [B, N, D*(num_experts+1)]
        return self.fusion(concat)  # [B, N, D]


# 直接修改？
class CTViT(nn.Module):
    def __init__(
        self,
        *,
        dim,   # 特征维度，每个token的向量长度
        codebook_size, 
        image_size,
        patch_size,
        temporal_patch_size,  # 时间patch的大小（多少帧合成一个patch）
        spatial_depth,  # 空间Transformer的深度
        temporal_depth,  # 时间Transformer的深度
        discr_base_dim = 16,
        dim_head = 64,
        heads = 8,
        channels = 1,
        use_vgg_and_gan = True,
        vgg = None,
        discr_attn_res_layers = (16,),
        use_hinge_loss = True,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size  # 每个patch包含多少帧（第三轴）

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)
        # 空间相对位置编码，返回偏置tensor
        # 这个偏置会在注意力计算中被加到注意力分数上，帮助模型理解patch之间的相对位置关系
        '''
        ContinuousPositionBias 是一个可学习的相对位置编码模块，通常实现为一个小的MLP或查找表。
        它根据 patch 之间的相对坐标（如横纵距离）输出一个偏置值，直接加到注意力分数上。
        这样，模型在计算注意力时会自动考虑空间结构。
        '''

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )
        # 将第一帧图像转换为patch embeddings
        # Rearrange 将输入张量的形状从 (b, c, 1, h, w) 转换为 (b, 1, h, w, c * p1 * p2)

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )
        # 3D切成patch

        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)
        self.vq = VectorQuantize(dim = dim, codebook_size = codebook_size, use_cosine_sim = True)
        # 把连续特征变成有限的“离散token”
        '''
        3. 工作原理
        准备一个codebook：里面有N个可学习的原型向量（每个向量维度为dim）。
        输入一个特征向量：比如经过Transformer编码后的patch特征。
        查找最近的codebook向量：计算输入向量与所有codebook向量的距离（如余弦距离或欧氏距离）。
        找到距离最近的那个，把输入向量替换成它。
        输出：离散化后的向量（即codebook中的向量），以及它的索引（token id）。
        '''

        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1 = patch_height, p2 = patch_width)
        )
        # 像素还原模块
        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
        )
        
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss
        # 生成器损失函数，使用hinge loss或bce loss
        # hinge loss 是一种常用的生成对抗网络损失函数，旨在最大化生成样本的分数，同时最小化真实样本的分数
        

    def calculate_video_token_mask(self, videos, video_frame_mask):
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        first_frame_mask, rest_frame_mask = video_frame_mask[:, :1], video_frame_mask[:, 1:]
        rest_vq_mask = rearrange(rest_frame_mask, 'b (f p) -> b f p', p = self.temporal_patch_size)
        video_mask = torch.cat((first_frame_mask, rest_vq_mask.any(dim = -1)), dim = -1)
        return repeat(video_mask, 'b f -> b (f hw)', hw = (h // ph) * (w // pw))

    def get_video_patch_shape(self, num_frames, include_first_frame = True):
        patch_frames = 0

        if include_first_frame:
            num_frames -= 1
            patch_frames += 1

        patch_frames += (num_frames // self.temporal_patch_size)

        return (patch_frames, *self.patch_height_width)

    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def frames_per_num_tokens(self, num_tokens):
        tokens_per_frame = self.image_num_tokens

        assert (num_tokens % tokens_per_frame) == 0, f'number of tokens must be divisible by number of tokens per frame {tokens_per_frame}'
        assert (num_tokens > 0)

        pseudo_frames = num_tokens // tokens_per_frames
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    def num_tokens_per_frames(self, num_frames, include_first_frame = True):
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        if include_first_frame:
            num_frames -= 1
            total_tokens += image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        return total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens

    def copy_for_eval(self):
        device = next(self.parameters()).device
        device=torch.device('cuda')
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    #@remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    #@remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]
        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    # 主要功能是 patch特征的空间-时间双重Transformer编码，输出每个patch融合时空上下文的高级特征
    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width  # 一共几个patch，如何切分的

        video_shape = tuple(tokens.shape[:-1])
        # tokens除了最后一个维度外的形状，通常是(batch, time, height, width)

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        device=torch.device('cuda')
        attn_bias = self.spatial_rel_pos_bias(h, w, device = device)
        # ：为空间Transformer的注意力机制生成空间相对位置偏置（Spatial Relative Position Bias）张量。
        # size为[heads, h*w, h*w]，【注意力头数，patch的总数】
        # 表示当前head下，第i个patch与第j个patch之间的相对位置关系。[head,i,j]

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # encode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        return tokens

    def decode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        video_shape = tuple(tokens.shape[:-1])

        # decode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.dec_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        # decode - spatial

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        device=torch.device('cuda')
        attn_bias = self.spatial_rel_pos_bias(h, w, device = device)

        tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # to pixels

        #first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]

        #first_frame = self.to_pixels_first_frame(first_frame_token)

        #rest_frames = self.to_pixels(rest_frames_tokens)

        recon_video = self.to_pixels(tokens)

        #recon_video = torch.cat((first_frame, rest_frames), dim = 2)

        return recon_video

    def forward(
        self,
        video,
        mask = None,
        return_recons = False,
        return_recons_only = False,
        return_discr_loss = False,
        apply_grad_penalty = True,
        return_only_codebook_ids = False,
        return_encoded_tokens=False
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4
        # 如果是图像，维度为4，否则为5（视频）
        #print(video.shape)

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device
        device=torch.device('cuda')
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # derive patches

        #first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        #rest_frames_tokens = self.to_patch_emb(rest_frames)
        #tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)
        tokens = self.to_patch_emb(video)
        # save height and width in

        shape = tokens.shape
        *_, h, w, _ = shape

        # encode - spatial
        tokens = self.encode(tokens)
        # 是在对patch特征进行空间和时间上的Transformer编码，提取更高级的时空特征。
        # 也就是直接进过CTVIT-ENCODER部分，然后直接量化输出
        # 这里的tokens是经过空间Transformer编码后的特征，
        # 形状为(batch, time, height, width, dim)

        # quantize
        tokens, packed_fhw_shape = pack([tokens], 'b * d')
        # 的作用是将多维patch特征展平成二维张量，方便后续量化（vector quantization）操作，
        # 并记录原始形状信息以便还原。

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)

        tokens, indices, commit_loss = self.vq(tokens, mask = vq_mask)
        # 量化操作将连续的patch特征转换为离散的codebook向量，
        # 并返回对应的codebook索引（indices）和重构损失（commit_loss）。
        # 量化后的tokens形状为(batch, time, height, width, dim)，
        # indices形状为(batch, time, height * width)，
        # commit_loss是一个标量，表示量化损失。

        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        if return_encoded_tokens:
            return tokens
            
        recon_video = self.decode(tokens)

        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w') if is_image else recon_video.clone()

        if return_recons_only:
            return returned_recon

        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(video, recon_video, reduction = 'none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c = c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(video, recon_video)

        # prepare a random frame index to be chosen for discriminator and perceptual loss

        pick_frame_logits = torch.randn(b, f)

        if exists(mask):
            mask_value = -torch.finfo(pick_frame_logits.dtype).max
            pick_frame_logits = pick_frame_logits.masked_fill(~mask, mask_value)

        frame_indices = pick_frame_logits.topk(1, dim = -1).indices

        # whether to return discriminator loss

        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            video = pick_video_frame(video, frame_indices)
            recon_video = pick_video_frame(recon_video, frame_indices)

            recon_video = recon_video.detach()
            video.requires_grad_()

            transform = T.Compose([T.Resize(256)])

            recon_video = transform(recon_video)
            video = transform(video)

            
            #print("TEST")
            #print(recon_video.shape)


            recon_video_discr_logits, video_discr_logits = map(self.discr, (recon_video, video))

            discr_loss = self.discr_loss(recon_video_discr_logits, video_discr_logits)

            if apply_grad_penalty:
                gp = gradient_penalty(video, video_discr_logits)
                loss = discr_loss + gp

            if return_recons:
                return loss, returned_recon

            return loss

        # early return if training on grayscale

        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, returned_recon

            return recon_loss

        # perceptual loss

        input_vgg_input = pick_video_frame(video, frame_indices)
        recon_vgg_input = pick_video_frame(recon_video, frame_indices)
        transform = T.Compose([T.Resize(256)])
        input_vgg_input = transform(input_vgg_input)
        recon_vgg_input=transform(recon_vgg_input)
        
        # handle grayscale for vgg
        
        if video.shape[1] == 1:
            input_vgg_input2, recon_vgg_input2 = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (input_vgg_input, recon_vgg_input))
        transform = T.Compose([T.Resize(256)])
        input_vgg_input2 = transform(input_vgg_input2)
        recon_vgg_input2 = transform(recon_vgg_input2)



        input_vgg_feats = self.vgg(input_vgg_input2)
        recon_vgg_feats = self.vgg(recon_vgg_input2)

        perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)

        # generator loss

        gen_loss = self.gen_loss(self.discr(recon_vgg_input))

        # calculate adaptive weight

        last_dec_layer = self.to_pixels[0].weight

        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max = 1e4)

        # combine losses

        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss

        if return_recons:
            return loss, returned_recon

        return loss
