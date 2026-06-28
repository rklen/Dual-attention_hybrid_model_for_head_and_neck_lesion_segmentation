

class ConvBlock(nn.Module):
    """2×(Conv3×3 + BN + ReLU)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1= nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.conv2= nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# ---------- Simple U-Net encoder (no EfficientNet) ----------
class Stem(nn.Module):
    """Stride-2 stem so f0 is at H/2 to match your old shapes."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.stem(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class SimpleUNetEncoder(nn.Module):
    """
    Emits four features with strides: f0 @ H/2, f1 @ H/4, f2 @ H/8, f3 @ H/16
    Channels: base, 2×base, 4×base, 8×base
    """
    def __init__(self, in_channels=1, base_ch=64):
        super().__init__()
        self.stem  = Stem(in_channels, base_ch)
        self.down1 = Down(base_ch,   base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
    def forward(self, x):
        f0 = self.stem(x)   # H/2
        f1 = self.down1(f0) # H/4
        f2 = self.down2(f1) # H/8
        f3 = self.down3(f2) # H/16
        return f0, f1, f2, f3


# ---------- Cross CBAM (only channel) ----------
class CrossCBAM(nn.Module):
    def __init__(self, c_high, c_low, reduction=16,
                 residual=True, init_gamma=0.0):
        super().__init__()
        self.residual = residual
        mid = max(1, c_high // reduction)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(c_high, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, c_high, 1, bias=False)
        )
        self.proj_low = nn.Conv2d(c_low, c_high, 1, bias=False)
        self.gamma_c = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x_high, x_low):
        H, W = x_high.shape[-2:]
        low = F.interpolate(x_low, size=(H, W), mode='bilinear', align_corners=False)
        low = self.proj_low(low)

        avg = F.adaptive_avg_pool2d(x_high, 1) + F.adaptive_avg_pool2d(low, 1)
        mx  = F.adaptive_max_pool2d(x_high, 1) + F.adaptive_max_pool2d(low, 1)
        ca  = torch.sigmoid(self.ca_mlp(avg) + self.ca_mlp(mx))
        x   = x_high * (1.0 + self.gamma_c * ca) if self.residual else x_high * ca
        return x


# ---------- Swin-style window attention helpers ----------

def window_partition(x, window_size):
    """
    x: (B, H, W, C)
    return: (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    windows: (num_windows*B, window_size*window_size, C)
    return: (B, H, W, C)
    """
    B = int(windows.shape[0] // (H * W / window_size / window_size))
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)
    return x

class WindowAttention(nn.Module):
    """Standard multi-head self-attention inside local windows."""
    def __init__(self, dim, window_size=7, num_heads=4, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x: (num_windows*B, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (3, B_, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinBlock(nn.Module):
    """
    Single Swin-like block (no shift, no relative bias) on (B,C,H,W).
    """
    def __init__(self, dim, num_heads=4, window_size=7, mlp_ratio=4.0,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size,
            num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "H and W must be divisible by window_size"

        # to (B,H,W,C)
        x_perm = x.permute(0, 2, 3, 1).contiguous()

        # window self-attn
        shortcut = x_perm
        x_norm = self.norm1(x_perm)
        windows = window_partition(x_norm, ws)        # (nW*B, ws*ws, C)
        attn_windows = self.attn(windows)             # (nW*B, ws*ws, C)
        x_merged = window_reverse(attn_windows, ws, H, W)  # (B,H,W,C)
        x_perm = shortcut + x_merged

        # MLP
        shortcut2 = x_perm
        x_norm2 = self.norm2(x_perm)
        x_mlp = self.mlp(x_norm2)
        x_out = shortcut2 + x_mlp   # (B,H,W,C)

        # back to (B,C,H,W)
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        return x_out


# ---------- Adjacent KV + Swin bottleneck ----------

class AdjacentKVBottleneck(nn.Module):
    """
    For each level i, gate f_i using cross-attention with K,V from f_{i+1}.
    Deepest f3 is refined by a Swin Transformer block; for level-3 gate,
    Q comes from original f3 and K,V come from Swin-refined f3.
    """
    def __init__(self, in_chs=(64,128,256,512), embed_dim=256,
                 kv_reduction=1, dropout=0.0,
                 self_attn_last=True,    # keep option for deepest gate
                 use_cbam=True, cbam_reduction=16, cbam_residual=True, cbam_init_gamma=0.0,
                 use_swin=True, swin_heads=4, swin_window_size=7):
        super().__init__()
        c0, c1, c2, c3 = in_chs
        self.E = embed_dim
        self.drop = nn.Dropout(dropout)
        self.kv_reduction = kv_reduction
        self.self_attn_last = self_attn_last
        self.use_cbam = use_cbam
        self.use_swin = use_swin

        # Q projections (per level)
        self.q0 = nn.Conv2d(c0, embed_dim, 1, bias=False)
        self.q1 = nn.Conv2d(c1, embed_dim, 1, bias=False)
        self.q2 = nn.Conv2d(c2, embed_dim, 1, bias=False)
        self.q3 = nn.Conv2d(c3, embed_dim, 1, bias=False)

        stride = kv_reduction if kv_reduction > 1 else 1

        # K,V for pairs: from next deeper level
        self.k1 = nn.Conv2d(c1, embed_dim, kernel_size=1, stride=stride, bias=False)
        self.v1 = nn.Conv2d(c1, embed_dim, kernel_size=1, stride=stride, bias=False)

        self.k2 = nn.Conv2d(c2, embed_dim, kernel_size=1, stride=stride, bias=False)
        self.v2 = nn.Conv2d(c2, embed_dim, kernel_size=1, stride=stride, bias=False)

        self.k3 = nn.Conv2d(c3, embed_dim, kernel_size=1, stride=stride, bias=False)
        self.v3 = nn.Conv2d(c3, embed_dim, kernel_size=1, stride=stride, bias=False)

        # self-attn for deepest (using Swin-refined f3 for K,V; Q from original f3)
        self.k3_self = nn.Conv2d(c3, embed_dim, 1, bias=False)
        self.v3_self = nn.Conv2d(c3, embed_dim, 1, bias=False)

        # project attended context back to each level's channels (as a gate)
        self.o0 = nn.Conv2d(embed_dim, c0, 1, bias=False)
        self.o1 = nn.Conv2d(embed_dim, c1, 1, bias=False)
        self.o2 = nn.Conv2d(embed_dim, c2, 1, bias=False)
        self.o3 = nn.Conv2d(embed_dim, c3, 1, bias=False)

        # learnable strengths (residual gating), start near identity
        self.gamma0 = nn.Parameter(torch.tensor(0.01))
        self.gamma1 = nn.Parameter(torch.tensor(0.01))
        self.gamma2 = nn.Parameter(torch.tensor(0.01))
        self.gamma3 = nn.Parameter(torch.tensor(0.01))

        # Cross-scale CBAM
        if use_cbam:
            self.cb01 = CrossCBAM(c_high=c0, c_low=c1,
                                  reduction=cbam_reduction, residual=cbam_residual,
                                  init_gamma=cbam_init_gamma)
            self.cb12 = CrossCBAM(c_high=c1, c_low=c2,
                                  reduction=cbam_reduction, residual=cbam_residual,
                                  init_gamma=cbam_init_gamma)
            self.cb23 = CrossCBAM(c_high=c2, c_low=c3,
                                  reduction=cbam_reduction, residual=cbam_residual,
                                  init_gamma=cbam_init_gamma)

        # refinement (depthwise separable conv per scale)
        self.ref0 = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1, groups=c0),
            nn.BatchNorm2d(c0),
            nn.GELU(),
            nn.Conv2d(c0, c0, 1))
        self.ref1 = nn.Sequential(
            nn.Conv2d(c1, c1, 3, padding=1, groups=c1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, 1))
        self.ref2 = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c2, 1))
        self.ref3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, padding=1, groups=c3),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, c3, 1))

        # Swin bottleneck on deepest feature f3
        if self.use_swin:
            self.swin = SwinBlock(
                dim=c3,
                num_heads=swin_heads,
                window_size=swin_window_size
            )

    def _attn_gate(self, q, K, V, out_proj):
        B, E, Hq, Wq = q.shape
        Sq = Hq * Wq
        qf = q.view(B, E, Sq).transpose(1, 2).contiguous()          # (B, Sq, E)

        Bk, Ek, Hk, Wk = K.shape
        assert Bk == B and Ek == E
        Sk = Hk * Wk
        kf = K.view(B, E, Sk).transpose(1, 2).contiguous()          # (B, Sk, E)
        vf = V.view(B, E, Sk).transpose(1, 2).contiguous()          # (B, Sk, E)

        k_transposed = kf.transpose(1, 2).contiguous()              # (B, E, Sk)

        attn = torch.matmul(qf, k_transposed) / math.sqrt(E)  # (B, Sq, Sk)
        attn = torch.softmax(attn, dim=-1)

        ctx  = torch.matmul(self.drop(attn), vf)       # (B, Sq, E)
        ctx  = ctx.transpose(1, 2).contiguous().view(B, E, Hq, Wq)
        gate = torch.sigmoid(out_proj(ctx))
        return gate

    def forward(self, f0, f1, f2, f3):
        # keep original f3 for Q
        f3_orig = f3

        # Swin bottleneck refinement on deepest level
        if self.use_swin:
            f3 = self.swin(f3)

        # cross-attention for upper levels
        g2 = f2 * (1 + self.gamma2 * self._attn_gate(self.q2(f2), self.k3(f3), self.v3(f3), self.o2))
        g1 = f1 * (1 + self.gamma1 * self._attn_gate(self.q1(f1), self.k2(g2), self.v2(g2), self.o1))
        g0 = f0 * (1 + self.gamma0 * self._attn_gate(self.q0(f0), self.k1(g1), self.v1(g1), self.o0))

        # deepest level self-attn: Q from original f3, K,V from (Swin-)refined f3
        if self.self_attn_last:
            K3 = self.k3_self(f3)
            V3 = self.v3_self(f3)
            g3 = f3 * (1.0 + self.gamma3 * self._attn_gate(self.q3(f3_orig), K3, V3, self.o3))
        else:
            g3 = f3

        # pairwise CBAM
        if self.use_cbam:
            g0 = self.cb01(g0, g1)
            g1 = self.cb12(g1, g2)
            g2 = self.cb23(g2, g3)
        # if self.use_cbam:
        #         h0 = self.cb01(f0, f1)  # high=f0, low=f1
        #         h1 = self.cb12(f1, f2)
        #         h2 = self.cb23(f2, f3)
        # else:
        #         h0, h1, h2 = f0, f1, f2

        # g2 = h2 * (1 + self.gamma2 * self._attn_gate(self.q2(h2), self.k3(f3), self.v3(f3), self.o2))
        # g1 = h1 * (1 + self.gamma1 * self._attn_gate(self.q1(h1), self.k2(g2), self.v2(g2), self.o1))
        # g0 = h0 * (1 + self.gamma0 * self._attn_gate(self.q0(h0), self.k1(g1), self.v1(g1), self.o0))
        # if self.self_attn_last:
        #     K3 = self.k3_self(f3)
        #     V3 = self.v3_self(f3)
        #     g3 = f3 * (1.0 + self.gamma3 * self._attn_gate(self.q3(f3_orig), K3, V3, self.o3))
        # else:
        #     g3 = f3

        # refinement
        ref0 = self.ref0(g0)
        ref1 = self.ref1(g1)
        ref2 = self.ref2(g2)
        ref3 = self.ref3(g3)

        return ref0, ref1, ref2, ref3


# ---------- Plain U-Net decoder ----------

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # Guard for odd spatial sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetDecoder(nn.Module):
    def __init__(self, chs=(64,128,256,512), n_classes=1):
        super().__init__()
        t0, t1, t2, t3 = chs
        self.up3 = Up(t3, t2, t2)  # H/16 → H/8
        self.up2 = Up(t2, t1, t1)  # H/8  → H/4
        self.up1 = Up(t1, t0, t0)  # H/4  → H/2
        self.up0 = nn.ConvTranspose2d(t0, t0, kernel_size=2, stride=2)  # H/2 → H
        self.head = nn.Conv2d(t0, n_classes, kernel_size=1)
        self.out2 = nn.Conv2d(t1, n_classes, kernel_size=1)
        self.out1 = nn.Conv2d(t0, n_classes, kernel_size=1)
    def forward(self, g0, g1, g2, g3):
        x = self.up3(g3, g2)
        x = self.up2(x, g1)
        out2_decs = self.out2(x)
        x = self.up1(x, g0)
        out1_decs = self.out1(x)
        x = self.up0(x)
        main_head = self.head(x)

        H, W = main_head.shape[2:]
        out1 = F.interpolate(out1_decs, (H, W), mode="bilinear", align_corners=False)
        out2 = F.interpolate(out2_decs, (H, W), mode="bilinear", align_corners=False)
        return main_head, out1, out2


# ---------- Full model ----------

class UNetSimple_AdjKV_CBAM(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, base_ch=64,
                 embed_dim=192, kv_reduction=2, dropout=0.0,
                 self_attn_last=True, use_cbam=True, cbam_reduction=16,
                 cbam_residual=True, cbam_init_gamma=0.0,
                 use_swin=True, swin_heads=4, swin_window_size=7):
        super().__init__()
        self.encoder = SimpleUNetEncoder(in_channels=in_channels, base_ch=base_ch)
        chs = (base_ch, base_ch*2, base_ch*4, base_ch*8)
        self.bottleneck = AdjacentKVBottleneck(
            in_chs=chs, embed_dim=embed_dim, kv_reduction=kv_reduction, dropout=dropout,
            self_attn_last=self_attn_last, use_cbam=use_cbam,
            cbam_reduction=cbam_reduction, cbam_residual=cbam_residual,
            cbam_init_gamma=cbam_init_gamma,
            use_swin=use_swin, swin_heads=swin_heads, swin_window_size=swin_window_size
        )
        self.decoder = UNetDecoder(chs=chs, n_classes=n_classes)

    def forward(self, x):
        f0, f1, f2, f3 = self.encoder(x)
        g0, g1, g2, g3 = self.bottleneck(f0, f1, f2, f3)
        return self.decoder(g0, g1, g2, g3)

import math
#  test
if __name__ == "__main__":
    mymodel = UNetSimple_AdjKV_CBAM(
        in_channels=1, n_classes=1,
        base_ch=32, embed_dim=96, kv_reduction=4,
        dropout=0.25, self_attn_last=True,
        use_cbam=True, cbam_reduction=4,
        cbam_residual=True, cbam_init_gamma=0.01,
        use_swin=True, swin_heads=4, swin_window_size=7
    )

