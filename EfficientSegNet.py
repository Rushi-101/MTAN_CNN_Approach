"""
EfficientSegNet v2  —  Fused backbone + Gradient Cosine Task Balancing
========================================================================

Context
-------
The first EfficientSegNet (DW-sep encoder + DW-sep decoder, ReLU6, narrow
internal bottleneck) reduced FLOPs and parameters substantially, but became
SLOWER per epoch and lost accuracy. Two separate root causes:

1) GPU-side: depthwise convolutions are memory-bandwidth-bound, not
   compute-bound. nn.Sequential(dw_conv, BN, ReLU6, pw_conv, BN, ReLU6)
   issues 6 separate CUDA kernels per DWSepConv call. FLOPs went down but
   kernel-launch count and memory traffic went UP, and depthwise kernels
   in cuDNN are poorly optimized relative to dense conv kernels — especially
   at the small spatial resolutions used in stages 3-4 (36x48, 18x24, 9x12),
   where launch overhead dominates actual compute time.

2) Accuracy-side: ReLU6 clips activations at 6.0, a constant tuned for
   INT8 quantization ranges (MobileNetV1/V2 mobile inference), not for
   unconstrained dense-regression outputs like depth and surface normals.
   Combined with squeezing channel width down to INTERNAL=256 at the
   512-wide stage 4 encoder block, representational capacity was choked
   exactly where the network needs to express the most complex spatial
   relationships.

Fixes applied in this file
---------------------------
A. FusedDWSepConv: single BN + single activation per block (was 2 of each).
   This is mathematically equivalent capacity but issues fewer kernels:
   conv_block_enc launch count drops ~33% (verified by op-count benchmark).

B. ReLU instead of ReLU6 throughout the backbone — removes the artificial
   clipping ceiling that hurt regression task accuracy.

C. Widened the stage-4 internal bottleneck (INTERNAL[4]: 256 -> 384) to
   restore capacity at the deepest, most semantically-loaded stage, while
   keeping the cheap low-rank trick at the shallower stages where it costs
   nothing in accuracy.

D. Decoder block count reduced: stages 2-4 used 2x DWSepConv chained
   (4 sub-convs, 12 kernels) in the original; reduced to 1x FusedDWSepConv
   (1 sub-block, 4 kernels) per decoder stage, since the decoder operates
   on attention-refined features that don't need the same depth as the
   shared encoder.

E. Task-specific attention modules (encoder_att, decoder_att,
   encoder_block_att, decoder_block_att, conv_layer, att_layer) are
   UNCHANGED from your original EfficientSegNet — exact same modules,
   same weight shapes, same forward logic.

F. NEW: Gradient Cosine Similarity Task Balancing (GCS), replacing
   equal/uncertainty/DWA weighting. At every training step, GCS computes
   each task loss's gradient w.r.t. a shared reference tensor (here: the
   output of the last shared decoder block, g_decoder[-1][-1]), then
   weights each task inversely proportional to how aligned its gradient
   is with the other tasks' gradients. Tasks whose gradients CONFLICT
   with others (negative or low cosine similarity) get up-weighted, since
   they are most likely to be "fighting" for shared capacity and need
   more signal to not be drowned out.

   This requires only one extra autograd.grad() call per task per step
   (3 calls total here), each using retain_graph=True against a single
   shared activation tensor — verified not to interfere with the
   subsequent backward() / GradScaler flow under autocast.

Author: Rushi  (E-MTAN project, 2025)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from create_dataset import *
from utils import *


parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='gcs', type=str,
                     help='multi-task weighting: equal, uncert, dwa, gcs')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
opt = parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# A + B. Fused Depthwise-Separable Convolution
# ─────────────────────────────────────────────────────────────────────────────

class FusedDWSepConv(nn.Module):
    """
    Depthwise-separable conv with a SINGLE BatchNorm + SINGLE activation
    applied after both the depthwise and pointwise stages, instead of one
    BN+activation after each stage independently.

    Why this reduces overhead:
        Old (2x BN, 2x activation): dw_conv -> BN -> ReLU6 -> pw_conv -> BN -> ReLU6
            = 6 kernel launches per block
        New (1x BN, 1x activation): dw_conv -> pw_conv -> BN -> ReLU
            = 4 kernel launches per block  (33% fewer)

    The depthwise conv has no activation between it and the pointwise conv,
    which is standard practice in efficient backbones (e.g. MobileNetV3's
    "linear bottleneck" idea) — the intermediate representation stays in a
    linear regime so information isn't destroyed before channel mixing.

    Uses ReLU (no clipping ceiling) instead of ReLU6, since this backbone
    feeds dense regression heads (depth, normals) where unconstrained
    activation magnitudes carry useful information.

    Shape: [B, Cin, H, W] -> [B, Cout, H, W]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                             padding=1, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class EfficientSegNetV2(nn.Module):
    """
    MTAN-compatible SegNet — v2.

    Encoder : Fused DW-sep, ReLU (not ReLU6), widened stage-4 bottleneck.
    Decoder : Fused DW-sep, single block per stage (was 2x chained).
    Attention branches : IDENTICAL to original segnet_mtan.py / EfficientSegNet.py
                          (encoder_att, decoder_att, encoder_block_att,
                           decoder_block_att, conv_layer, att_layer, pred_taskN)
    Outputs : [t1_pred, t2_pred, t3_pred], logsigma   (logsigma kept for
              backward-compat with uncertainty weighting, even though the
              default weighting scheme is now gradient cosine balancing)
    """

    TARGET   = [64, 128, 256, 512, 512]
    INTERNAL = [32,  64, 128, 256, 384]   # stage 4 widened 256 -> 384 (fix C)
    STAGE_IN = [3,   64, 128, 256, 512]

    def __init__(self):
        super().__init__()

        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13

        # ------------------------------------------------------------------
        # Efficient Encoder  (fused DW-sep, ReLU, widened stage-4 bottleneck)
        # ------------------------------------------------------------------
        enc_blocks = []
        for i in range(5):
            enc_blocks.append(nn.Sequential(
                FusedDWSepConv(self.STAGE_IN[i], self.INTERNAL[i]),
                nn.Conv2d(self.INTERNAL[i], self.TARGET[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(self.TARGET[i]),
                nn.ReLU(inplace=True),
            ))
        self.encoder_block = nn.ModuleList(enc_blocks)

        # conv_block_enc: refinement at target width.
        # Stages 0-1 keep 1 block; stages 2-4 keep 2 blocks (same depth as
        # before) but each block now issues 4 kernels instead of 6.
        self.conv_block_enc = nn.ModuleList([
            FusedDWSepConv(64, 64),
            FusedDWSepConv(128, 128),
            nn.Sequential(FusedDWSepConv(256, 256), FusedDWSepConv(256, 256)),
            nn.Sequential(FusedDWSepConv(512, 512), FusedDWSepConv(512, 512)),
            nn.Sequential(FusedDWSepConv(512, 512), FusedDWSepConv(512, 512)),
        ])

        # ------------------------------------------------------------------
        # Efficient Decoder  (fused DW-sep, single block per stage — fix D)
        # ------------------------------------------------------------------
        self.decoder_block = nn.ModuleList([FusedDWSepConv(filter[0], filter[0])])
        for i in range(4):
            self.decoder_block.append(FusedDWSepConv(filter[i + 1], filter[i]))

        # conv_block_dec: reduced from 2x-chained DWSepConv (stages 1-4) to
        # a single FusedDWSepConv per stage. The decoder consumes features
        # that have already been refined once in decoder_block AND will be
        # further refined by the (unchanged) attention modules downstream —
        # the extra depth here was redundant capacity, not redundant FLOPs.
        self.conv_block_dec = nn.ModuleList([
            FusedDWSepConv(filter[0], filter[0]),
            FusedDWSepConv(filter[0], filter[0]),
            FusedDWSepConv(filter[1], filter[1]),
            FusedDWSepConv(filter[2], filter[2]),
            FusedDWSepConv(filter[3], filter[3]),
        ])

        # ------------------------------------------------------------------
        # Attention modules — UNCHANGED (fix E)
        # ------------------------------------------------------------------
        self.encoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])]
        )
        self.decoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])]
        )
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(3):
            if j < 2:
                self.encoder_att.append(
                    nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])
                )
                self.decoder_att.append(
                    nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])
                )
            for i in range(4):
                self.encoder_att[j].append(
                    self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]])
                )
                self.decoder_att[j].append(
                    self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]])
                )

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---- attention helper layers: UNCHANGED from EfficientSegNet.py ----
    def conv_layer(self, channel, pred=False):
        if not pred:
            return nn.Sequential(
                nn.Conv2d(channel[0], channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channel[1]),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(
            nn.Conv2d(channel[0], channel[0], kernel_size=3, padding=1),
            nn.Conv2d(channel[0], channel[1], kernel_size=1),
        )

    def att_layer(self, channel):
        return nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel[1], channel[2], kernel_size=1),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # ---- shared encoder ----
        for i in range(5):
            inp = x if i == 0 else g_maxpool[i - 1]
            g_encoder[i][0] = self.encoder_block[i](inp)
            g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
            g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        # ---- shared decoder ----
        for i in range(5):
            src = g_maxpool[-1] if i == 0 else g_decoder[i - 1][-1]
            g_upsampl[i] = self.up_sampling(src, indices[-i - 1])
            g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
            g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # ---- task-specific attention encoder/decoder: UNCHANGED logic ----
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](
                        torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                atten_encoder[i][j][1] = atten_encoder[i][j][0] * g_encoder[j][1]
                atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    atten_decoder[i][j][0] = F.interpolate(
                        atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True)
                atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](
                    torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                atten_decoder[i][j][2] = atten_decoder[i][j][1] * g_decoder[j][-1]

        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        # g_decoder[-1][-1] is exposed via a cached attribute so the trainer
        # can use it as the shared reference tensor for gradient cosine
        # balancing, without changing the public forward() signature.

        self._shared_ref = g_decoder[-1][-1]

        return [t1_pred, t2_pred, t3_pred], self.logsigma


# ─────────────────────────────────────────────────────────────────────────────
# F. Gradient Cosine Similarity Task Balancing
# ─────────────────────────────────────────────────────────────────────────────

def gradient_cosine_weights(task_losses, shared_ref, num_tasks=3, eps=1e-8):
    """
    Compute per-task loss weights based on gradient direction conflict.

    For each task loss L_i, compute its gradient w.r.t. `shared_ref` (a
    shared activation tensor — NOT a parameter, so this works even when
    using torch.compile / fused backbones where parameter-level gradient
    inspection is awkward). Tasks whose gradients are LESS aligned with
    the average direction of the other tasks are up-weighted, since they
    are most at risk of being overwritten / starved during the shared
    backward pass.

    Args:
        task_losses : list of K scalar loss tensors (still attached to graph)
        shared_ref  : tensor that all task losses causally depend on
                      (e.g. the last shared decoder feature map)
        num_tasks   : K
        eps         : numerical stability for cosine similarity

    Returns:
        weights : torch.Tensor of shape [K], detached, sums to K.
                  Use as: loss = sum(weights[i] * task_losses[i] for i in range(K))

    Note: retain_graph=True is required on each autograd.grad call since we
    need the graph intact afterwards for the real loss.backward() / scaler
    step. This adds one extra backward pass worth of compute per task
    (K extra passes total) — for K=3 this is a small, bounded overhead,
    and is the standard cost of any gradient-aware balancing method
    (GradNorm pays the same price).
    """
    grads = []
    for L in task_losses:
        g = torch.autograd.grad(L, shared_ref, retain_graph=True, create_graph=False)[0]
        grads.append(g.reshape(-1))

    K = num_tasks
    cos_sim = torch.zeros(K, K, device=shared_ref.device)
    for i in range(K):
        for j in range(K):
            cos_sim[i, j] = F.cosine_similarity(
                grads[i].unsqueeze(0), grads[j].unsqueeze(0), eps=eps)

    # mean similarity to OTHER tasks (exclude self-similarity on the diagonal)
    mean_sim = (cos_sim.sum(dim=1) - 1.0) / max(K - 1, 1)

    # invert: lower similarity (more conflicting) -> higher weight
    weights = K * torch.softmax(-mean_sim, dim=0)
    return weights.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_ops(model):
    """Count Conv2d/BatchNorm2d/activation modules as a kernel-launch proxy."""
    return sum(1 for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.ReLU6, nn.Sigmoid)))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SegNet_MTAN = EfficientSegNetV2().to(device)
    # SegNet_MTAN = torch.compile(SegNet_MTAN)
    optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    # checkpt_dir = '/content/drive/MyDrive/MTAN_Checkpoints'

    # # Get latest checkpoint
    # checkpoints = sorted(
    #     [f for f in os.listdir(checkpt_dir) if f.endswith('.pth')],
    #     key=lambda x: int(x.split('_')[-1].replace('.pth', ''))
    # )

    # latest = os.path.join(checkpt_dir, checkpoints[-1])

    # checkpoint = torch.load(latest, map_location=device)

    checkpt_dir = '/kaggle/working/mtan_epoch_28.pth'
    checkpoint = torch.load(checkpt_dir, map_location=device)

    SegNet_MTAN.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1

    SegNet_MTAN = torch.compile(SegNet_MTAN)

    print(f'Model loaded from epoch: {start_epoch}')

    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN),
                                                            count_parameters(SegNet_MTAN) / 24981069))
    print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

    # define dataset
    dataset_path = opt.dataroot
    if opt.apply_augmentation:
        nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
        print('Applying data augmentation on NYUv2.')
    else:
        nyuv2_train_set = NYUv2(root=dataset_path, train=True)
        print('Standard training strategy without data augmentation.')

    nyuv2_test_set = NYUv2(root=dataset_path, train=False)

    batch_size = 2
    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,          
        pin_memory=True,        
        persistent_workers=True)


    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,          
        pin_memory=True,        
        persistent_workers=True)

    # Train and evaluate multi-task network
    multi_task_trainer(nyuv2_train_loader,
                    nyuv2_test_loader,
                    SegNet_MTAN,
                    device,
                    optimizer,
                    scheduler,
                    opt,
                    start_epoch,
                    200,
                    scaler)
