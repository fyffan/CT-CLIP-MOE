import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from beartype import beartype
from typing import Tuple

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def l2norm(t):
    return F.normalize(t, dim = -1)

# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# PEG - position generating module

class PEG(nn.Module):
    def __init__(self, dim, causal = False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups = dim)
        # 使用3D卷积来处理时间和空间维度
        # groups = dim表示每个通道独立卷积
        # 什么叫每个通道单独卷积？
        # 这意味着每个通道的卷积核只作用于该通道的数据，不与其他通道共享权重。

    @beartype
    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        needs_shape = x.ndim == 3  
        # 判断输入维度是否为3
        assert not (needs_shape and not exists(shape))
        # 如果是3维输入，则必须提供shape参数

        orig_shape = x.shape

        if needs_shape:
            x = x.reshape(*shape, -1)
            # 如果输入是展平的3维张量（如 [batch, n, d]），
            # 则用 shape 参数（如 [batch, t, h, w]）还原为
            # 多维结构（如 [batch, t, h, w, d]）

        x = rearrange(x, 'b ... d -> b d ...')
        # 把最后一维（特征维 d）移到第二维，变成 [batch, d, ...]。
        # 这样做是为了适应3D卷积的输入格式。
        # 因为PyTorch的3D卷积要求输入格式为 [batch, channels, depth, height, width]。

        frame_padding = (2, 0) if self.causal else (1, 1)
        # 如果 self.causal == True（因果卷积，常用于时序建模），
        # 则只在“前面”pad 2，“后面”pad 0（即只看过去，不看未来）。

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value = 0.)
        # 在slice维度只pad前面两个
        x = self.dsconv(x)

        x = rearrange(x, 'b d ... -> b ... d')

        if needs_shape:
            x = rearrange(x, 'b ... d -> b (...) d')

        return x.reshape(orig_shape)
    # 输出是“加了可学习局部位置编码的特征”，
    # shape与输入完全一致，
    # 内容上每个token已具备局部时空感知能力。

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        num_null_kv = 0,
        norm_context = True,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale  # 缩放因子，通常为1/sqrt(dim_head)
        inner_dim = dim_head * heads  # 内部维度是头数乘以每个头的维度
        dim_context = default(dim_context, dim)  
        # 如果没有提供dim_context，则默认为dim

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads = heads)
            # 使用Alibi位置偏置来处理因果注意力

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 将输入特征（维度dim）映射为inner_dim = dim_head * heads，用于生成Query。
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)
        # 将上下文特征（维度dim_context）映射为inner_dim * 2，用于生成Key和Value。

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        # 可学习的缩放参数，用于Query和Key的缩放。

        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        # 还原成原始维度

    def forward(
        self,
        x,
        mask = None,  # 注意力掩码，形状为 [batch, seq_len]
        context = None,  # 上下文特征，形状为 [batch, context_len, dim_context]
        attn_bias = None  # 注意力偏置，形状为 [batch, heads, seq_len, context_len]
    ):
        batch, device, dtype = x.shape[0], x.device, x.dtype
        device=torch.device('cuda')
        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)
        # 如果没有提供上下文，则使用输入x作为Key和Value的输入。

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)
        # 把 kv_input 映射为 Key 和 Value，
        # shape [batch, n, 2*heads*dim_head]，再切分成 k 和 v

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b = batch, r = 2).unbind(dim = -2)
        # 可学习的 null key/value，扩展到 batch 维
        # 拼接到真实的 k/v 前面，增加“虚拟token”

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        # 对 q、k 做 L2 归一化
        # 乘以可学习缩放参数（每个 head 的每个维度独立缩放）

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # 计算 QK 的点积，得到注意力分数 [batch, heads, n, n_kv]
        # 乘以缩放因子

        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value = 0.)
            sim = sim + attn_bias
            # attn_bias通常是空间或时序的相对位置偏置，
            # shape 一般为 [heads, n, n_kv]。

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
            # 对 mask 的最后一维（key/value 维）前面补 self.num_null_kv 个 True（对应 null kv 的部分），
            # 保证 mask 和 sim 的 shape 对齐。
            # 调整 mask 的 shape，扩展到 [batch, 1, 1, seq_len_kv]，
            # 方便和 sim（[batch, heads, seq_len_q, seq_len_kv]）做广播。
            # 对于 mask 为 False 的位置（即不允许关注的位置），将 sim 的对应分数填成极小值（负无穷），
            # 这样 softmax 后这些位置的注意力权重就是0。

        if self.causal:
            sim = sim + self.rel_pos_bias(sim)
            device=torch.device('cuda')
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
            # 

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# alibi positional bias for extrapolation

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    def get_bias(self, i, j, device):
        device=torch.device('cuda')
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]
        device=torch.device('cuda')
        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 2, # 2 for images, 3 for video
        layers = 2,
        log_dist = True,
        cache_rel_pos = False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, *dimensions, device = torch.device('cpu')):

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            device=torch.device('cuda')
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.to(torch.float32)

        for layer in self.net:
            rel_pos = layer(rel_pos.float())

        return rearrange(rel_pos, 'i j h -> h i j')

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context = None,  # 交叉注意力时context的维度
        causal = False,  # 是否使用因果mask
        dim_head = 64,
        heads = 8,
        ff_mult = 4,  # 前馈网络扩展倍数
        peg = False,  # 是否使用PEG位置编码
        peg_causal = False,  # PEG位置编码是否使用因果mask
        attn_num_null_kv = 2,  # 注意力中null key/value的数量
        has_cross_attn = False,  # 是否有交叉注意力层
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PEG(dim = dim, causal = peg_causal) if peg else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout),
                Attention(dim = dim, dim_head = dim_head, dim_context = dim_context, heads = heads, causal = False, num_null_kv = attn_num_null_kv, dropout = attn_dropout) if has_cross_attn else None,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))
        # 每一层包含
        # 1. PEG位置编码（可选）
        # 2. 自注意力层
        # 3. 交叉注意力层（可选）
        # 4. 前馈网络

        self.norm_out = LayerNorm(dim)  # 输出层归一化

    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias = None,
        context = None,
        self_attn_mask = None,
        cross_attn_context_mask = None
    ):

        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape = video_shape) + x

            x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x

            x = ff(x) + x

        return self.norm_out(x)
