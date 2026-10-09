import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _gn(c):
    g = 8
    while c % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, c)


class LocalContrastNorm(nn.Module):
    def __init__(self, kernel_size=9, eps=1e-5):
        super().__init__()
        self.k = kernel_size
        self.p = kernel_size // 2
        self.eps = eps

    def forward(self, x):
        pad = F.pad(x, (self.p, self.p, self.p, self.p), mode='reflect')
        mean = F.avg_pool2d(pad, self.k, stride=1)
        sq = F.pad(x * x, (self.p, self.p, self.p, self.p), mode='reflect')
        mean_sq = F.avg_pool2d(sq, self.k, stride=1)
        var = (mean_sq - mean * mean).clamp(min=0)
        std = torch.sqrt(var + self.eps)
        return (x - mean) / std


class PatchSelfSimilarity(nn.Module):
    def __init__(self, in_channels, patch_size=5, neighbor_size=5, dilation=1):
        super().__init__()
        self.patch_size = patch_size
        self.neighbor_size = neighbor_size
        self.dilation = dilation
        self.padding = dilation * (neighbor_size - 1) // 2
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            _gn(in_channels),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Parameter(
            torch.zeros(1, neighbor_size * neighbor_size, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        P, K = self.patch_size, self.neighbor_size

        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect') if (pad_h or pad_w) else x
        _, _, Hp, Wp = x_pad.shape

        unf = F.unfold(x_pad, kernel_size=P, stride=P)
        pH, pW = Hp // P, Wp // P
        unf = unf.view(B, C, P * P, -1)
        patch_feat = unf.mean(dim=2).view(B, C, pH, pW)

        pf = self.patch_encoder(patch_feat)
        nb = F.unfold(
            pf, kernel_size=K, dilation=self.dilation, padding=self.padding
        ).view(B, C, K * K, pH, pW)

        cn = F.normalize(pf.unsqueeze(2), dim=1)
        un = F.normalize(nb, dim=1)
        sim = (cn * un).sum(dim=1)
        w = F.softmax(self.weight, dim=1)
        sim_map_patch = (sim * w).sum(dim=1, keepdim=True)

        sim_map = F.interpolate(
            sim_map_patch, size=(Hp, Wp), mode='bilinear', align_corners=False)
        if pad_h or pad_w:
            sim_map = sim_map[:, :, :H, :W]
        return sim_map


class MultiScalePatchSelfSimilarity(nn.Module):
    def __init__(self, in_channels,
                 patch_sizes=(5, 7, 9),
                 neighbor_sizes=(5, 7, 9),
                 dilations=(1, 2)):
        super().__init__()
        assert len(patch_sizes) == len(neighbor_sizes)
        self.sim_modules = nn.ModuleList([
            PatchSelfSimilarity(in_channels, ps, ns, d)
            for ps, ns in zip(patch_sizes, neighbor_sizes)
            for d in dilations
        ])
        self.n_scales = len(patch_sizes) * len(dilations)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.n_scales, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        maps = [m(x) for m in self.sim_modules]
        cat = torch.cat(maps, dim=1)
        return cat, self.fusion(cat)


class SharedFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feat_dim=64):
        super().__init__()
        self.lcn = LocalContrastNorm(kernel_size=9)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            _gn(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, feat_dim, 3, padding=1, bias=False),
            _gn(feat_dim), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=2, dilation=2, bias=False),
            _gn(feat_dim), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=4, dilation=4, bias=False),
            _gn(feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(self.lcn(x))


class CrossDiffEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels=None, patch_size=5):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, 1, bias=False),
            _gn(mid_channels), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels + 5, mid_channels, 3, padding=1, bias=False),
            _gn(mid_channels), nn.ReLU(inplace=True),
        )

    def _struct_diff(self, f1, f2):
        B, C, H, W = f1.shape
        P = self.patch_size
        pad = P // 2
        p1 = F.unfold(f1, P, padding=pad).view(B, C, P * P, H, W)
        p2 = F.unfold(f2, P, padding=pad).view(B, C, P * P, H, W)
        m1, m2 = p1.mean(dim=2), p2.mean(dim=2)
        v1 = p1.var(dim=2, unbiased=False)
        v2 = p2.var(dim=2, unbiased=False)
        d_mean = torch.abs(m1 - m2).mean(dim=1, keepdim=True)
        d_var = torch.abs(v1 - v2).mean(dim=1, keepdim=True)
        return d_mean, d_var

    def forward(self, f1, f2):
        eps = 1e-6
        d_abs = torch.abs(f1 - f2).mean(dim=1, keepdim=True)
        d_log = torch.abs(
            torch.log((f1.abs() + eps) / (f2.abs() + eps))
        ).mean(dim=1, keepdim=True)
        n1 = F.normalize(f1, dim=1)
        n2 = F.normalize(f2, dim=1)
        d_corr = (1 - (n1 * n2).sum(dim=1, keepdim=True)) * 0.5

        d_abs = d_abs / (d_abs.amax(dim=(2, 3), keepdim=True) + eps)
        d_log = d_log / (d_log.amax(dim=(2, 3), keepdim=True) + eps)

        d_mean, d_var = self._struct_diff(f1, f2)
        d_mean = d_mean / (d_mean.amax(dim=(2, 3), keepdim=True) + eps)
        d_var = d_var / (d_var.amax(dim=(2, 3), keepdim=True) + eps)

        diffs = torch.cat([d_abs, d_log, d_corr, d_mean, d_var], dim=1)
        z = self.proj(torch.cat([f1, f2], dim=1))
        z = self.fuse(torch.cat([z, diffs], dim=1))
        return z, diffs


class CrossPatchMatcher(nn.Module):
    def __init__(self, in_channels, patch_size=5, neighbor_size=7, dilation=1):
        super().__init__()
        self.p = patch_size
        self.k = neighbor_size
        self.dil = dilation
        self.pad = dilation * (neighbor_size - 1) // 2
        self.q = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.k_proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.weight = nn.Parameter(
            torch.zeros(1, neighbor_size * neighbor_size, 1, 1))

    def _to_patch(self, f):
        B, C, H, W = f.shape
        P = self.p
        ph = (P - H % P) % P
        pw = (P - W % P) % P
        if ph or pw:
            f = F.pad(f, (0, pw, 0, ph), mode='reflect')
        Hp, Wp = f.shape[2], f.shape[3]
        u = F.unfold(f, P, stride=P).view(B, C, P * P, -1).mean(dim=2)
        return u.view(B, C, Hp // P, Wp // P)

    def _match(self, qf, kf):
        B, C, pH, pW = qf.shape
        K = self.k
        nb = F.unfold(
            kf, kernel_size=K, dilation=self.dil, padding=self.pad
        ).view(B, C, K * K, pH, pW)
        cn = F.normalize(qf.unsqueeze(2), dim=1)
        un = F.normalize(nb, dim=1)
        sim = (cn * un).sum(dim=1)
        w = F.softmax(self.weight, dim=1)
        sim = (sim * w).sum(dim=1, keepdim=True)
        return (sim + 1) * 0.5

    def forward(self, f1, f2):
        p1 = self._to_patch(self.q(f1))
        p2 = self._to_patch(self.k_proj(f2))
        sim_fw = self._match(p1, p2)
        sim_bw = self._match(p2, p1)
        B, _, H, W = f1.shape
        sim_fw = F.interpolate(sim_fw, size=(H, W), mode='bilinear', align_corners=False)
        sim_bw = F.interpolate(sim_bw, size=(H, W), mode='bilinear', align_corners=False)
        return sim_fw, sim_bw


def compute_handcraft_di(x1, x2, eps=1e-6):
    d_abs = (x1 - x2).abs().mean(dim=1, keepdim=True)
    d_log = torch.abs(
        torch.log((x1.abs() + eps) / (x2.abs() + eps))
    ).mean(dim=1, keepdim=True)
    n1, n2 = F.normalize(x1, dim=1), F.normalize(x2, dim=1)
    d_corr = (1 - (n1 * n2).sum(dim=1, keepdim=True)) * 0.5

    def local_mean(x, k=5):
        return F.avg_pool2d(x, k, stride=1, padding=k // 2)

    d_mean = (local_mean(x1) - local_mean(x2)).abs().mean(dim=1, keepdim=True)

    def _norm(t):
        tmin = t.amin(dim=(2, 3), keepdim=True)
        tmax = t.amax(dim=(2, 3), keepdim=True)
        return (t - tmin) / (tmax - tmin + eps)

    hand_prob = (_norm(d_abs) + _norm(d_log) + _norm(d_corr) + _norm(d_mean)) / 4.0
    hand_prob = hand_prob.clamp(1e-4, 1 - 1e-4)
    hand_logit = torch.logit(hand_prob)
    return hand_prob, hand_logit


class SinglePassDiff(nn.Module):
    def __init__(self, feat_dim, patch_size=5, neighbor_size=7):
        super().__init__()
        self.diff_encoder = CrossDiffEncoder(
            feat_dim, mid_channels=feat_dim, patch_size=patch_size)
        self.matcher = CrossPatchMatcher(
            feat_dim, patch_size=patch_size, neighbor_size=neighbor_size)
        self.residual_head = nn.Sequential(
            nn.Conv2d(feat_dim + 4, 32, 3, padding=1, bias=False),
            _gn(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, f1, f2, x1, x2):
        z, diffs = self.diff_encoder(f1, f2)
        struct_diff = diffs[:, 3:5]
        sim_fw, sim_bw = self.matcher(f1, f2)
        sim_cat = torch.cat([sim_fw, sim_bw], dim=1)

        _, hand_logit = compute_handcraft_di(x1, x2)

        residual = self.residual_head(
            torch.cat([z, sim_cat, struct_diff], dim=1))

        di_logit = hand_logit + self.residual_scale * residual

        return di_logit, sim_fw, sim_bw, struct_diff, z, hand_logit


class SegHead(nn.Module):
    def __init__(self, feat_dim, di_enh_channels=5):
        super().__init__()
        in_ch = feat_dim + di_enh_channels + 1 + 1 + 2
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            _gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, feat, di_enh, sim_fw, sim_bw, struct_diff):
        x = torch.cat([feat, di_enh, sim_fw, sim_bw, struct_diff], dim=1)
        return self.net(x)


class RCRL(nn.Module):
    def __init__(self, in_channels=3, feat_dim=64,
                 patch_size=5, neighbor_size=7,
                 use_multiscale_ss=True,
                 ss_patch_sizes=(5, 7, 9),
                 ss_dilations=(1, 2),
                 n_iter=None):
        super().__init__()
        self.use_multiscale_ss = use_multiscale_ss
        self.extractor = SharedFeatureExtractor(in_channels, feat_dim)

        if use_multiscale_ss:
            n_scales = len(ss_patch_sizes) * len(ss_dilations)
            self.n_scales = n_scales
            self.mspss = MultiScalePatchSelfSimilarity(
                feat_dim,
                patch_sizes=ss_patch_sizes,
                neighbor_sizes=ss_patch_sizes,
                dilations=ss_dilations,
            )
            self.ss_fuse = nn.Sequential(
                nn.Conv2d(3 * n_scales, 32, 3, padding=1, bias=False),
                _gn(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 4, 1),
            )
            self.di_enh_ch = 5
        else:
            self.n_scales = 0
            self.mspss = None
            self.ss_fuse = None
            self.di_enh_ch = 1

        self.diff_pass = SinglePassDiff(
            feat_dim, patch_size=patch_size, neighbor_size=neighbor_size)
        self.seg_head = SegHead(feat_dim, di_enh_channels=self.di_enh_ch)

    def forward(self, x1, x2):
        f1 = self.extractor(x1)
        f2 = self.extractor(x2)

        struct_ss = None
        if self.use_multiscale_ss:
            ss1_cat, _ = self.mspss(f1)
            ss2_cat, _ = self.mspss(f2)
            ss_diff = torch.abs(ss1_cat - ss2_cat)
            struct_ss = self.ss_fuse(
                torch.cat([ss1_cat, ss2_cat, ss_diff], dim=1))

        di_logit, sim_fw, sim_bw, struct_diff, z, hand_logit = \
            self.diff_pass(f1, f2, x1, x2)
        di_prob = torch.sigmoid(di_logit)
        hand_prob = torch.sigmoid(hand_logit)

        if struct_ss is not None:
            di_prob_enh = torch.cat([di_prob, struct_ss], dim=1)
        else:
            di_prob_enh = di_prob

        logits = self.seg_head(z, di_prob_enh, sim_fw, sim_bw, struct_diff)

        return {
            'logits': logits,
            'di_prob': di_prob,
            'di_logits': di_logit,
            'hand_prob': hand_prob,
            'hand_logits': hand_logit,
            'residual_scale': self.diff_pass.residual_scale.detach(),  # ★ 新增
            'f1': f1,
            'f2': f2,
            'sim_fw': sim_fw,
            'sim_bw': sim_bw,
            'struct_diff': struct_diff,
            'z': z,
            'struct_ss': struct_ss,
        }


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logit, target):
        if logit.dim() == 4 and logit.size(1) == 1:
            logit = logit[:, 0]
        if target.dim() == 4:
            target = target[:, 0]
        prob = torch.sigmoid(logit).reshape(logit.size(0), -1)
        tgt = target.reshape(target.size(0), -1).float()
        inter = (prob * tgt).sum(dim=1)
        union = prob.sum(dim=1) + tgt.sum(dim=1)
        dice = (2 * inter + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class UnsupervisedLoss(nn.Module):

    def __init__(self,
                 w_pseudo=1.0,
                 w_di=0.5,
                 w_patch_self=0.3,
                 n_iter=None):
        super().__init__()
        self.w_pseudo = w_pseudo
        self.w_di = w_di
        self.w_patch_self = w_patch_self
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    @staticmethod
    def _ensure_4d(pseudo):
        return pseudo.unsqueeze(1) if pseudo.dim() == 3 else pseudo

    def _seg_loss(self, logits, pseudo):
        return self.bce(logits, pseudo) + self.dice(logits, pseudo)

    def _patch_self_consistency(self, sim_fw, sim_bw, pseudo):

        w = (1 - pseudo).detach()
        loss_fw = (w * (1 - sim_fw) ** 2).mean()
        loss_bw = (w * (1 - sim_bw) ** 2).mean()
        return loss_fw + loss_bw

    def forward(self, out, pseudo):
        pseudo = self._ensure_4d(pseudo)

        loss = self.w_pseudo * self._seg_loss(out['logits'], pseudo)

        loss = loss + self.w_di * self._seg_loss(out['di_logits'], pseudo)

        loss = loss + self.w_patch_self * self._patch_self_consistency(
            out['sim_fw'], out['sim_bw'], pseudo)

        return loss


def otsu_threshold(x: np.ndarray, nbins: int = 256) -> float:
    x = x.astype(np.float32)
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return 0.5
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    hist, edges = np.histogram(x_norm, bins=nbins, range=(0, 1))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    centers = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.argmax(sigma_b))
    return float(centers[k])


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RCRL(
        in_channels=3, feat_dim=32,
        patch_size=3, neighbor_size=7,
        use_multiscale_ss=True,
        ss_patch_sizes=(5, 7, 9),
        ss_dilations=(1, 2),
    ).to(device)

    criterion = UnsupervisedLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x1 = torch.randn(2, 3, 256, 256).to(device)
    x2 = torch.randn(2, 3, 256, 256).to(device)
    pseudo = (torch.rand(2, 1, 256, 256) > 0.7).float().to(device)

    out = model(x1, x2)
    loss = criterion(out, pseudo)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"[Train-step] loss = {loss.item():.4f}")
    print(f"[Shapes] logits={tuple(out['logits'].shape)}, "
          f"di_prob={tuple(out['di_prob'].shape)}, "
          f"sim_fw={tuple(out['sim_fw'].shape)}, "
          f"struct_diff={tuple(out['struct_diff'].shape)}")
    print(f"[SegHead] in_ch = {model.seg_head.in_ch}")
    print(f"[Residual] scale = {out['residual_scale'].item():.6f}  (初始应≈0)")
    diff = (out['di_prob'] - out['hand_prob']).abs().mean().item()
    print(f"|di_prob - hand_prob| = {diff:.6e}  (初始应≈0)")

    model.eval()
    with torch.no_grad():
        xi1 = torch.randn(1, 3, 593, 921).to(device)
        xi2 = torch.randn(1, 3, 593, 921).to(device)
        out_i = model(xi1, xi2)
        di = out_i['di_prob'][0, 0].cpu().numpy()
        thr = otsu_threshold(di)
        binary = (di > thr).astype(np.uint8)
        print(f"[Inference 593x921] OTSU thr = {thr:.4f}, "
              f"change ratio = {binary.mean():.4f}")
    print("[OK] model.py self-test passed.")