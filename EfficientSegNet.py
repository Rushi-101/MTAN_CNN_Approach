import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from create_dataset import *
from utils import *

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
opt = parser.parse_args()


class DWSepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dw = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )

        self.pw = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class EfficientSegNet(nn.Module):
    """
    MTAN-compatible SegNet.

    Encoder:
        Efficient encoder from Efficient_MTAN.py

    Decoder:
        Fully replaced with DWSepConv blocks

    Attention branches:
        Identical to original segnet_mtan.py

    Outputs:
        [t1_pred, t2_pred, t3_pred], logsigma
    """

    TARGET = [64, 128, 256, 512, 512]
    INTERNAL = [32, 64, 128, 256, 256]
    STAGE_IN = [3, 64, 128, 256, 512]

    def __init__(self):
        super().__init__()

        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13

        # ------------------------------------------------------------------
        # Efficient Encoder
        # ------------------------------------------------------------------
        enc_blocks = []
        for i in range(5):
            enc_blocks.append(
                nn.Sequential(
                    DWSepConv(self.STAGE_IN[i], self.INTERNAL[i]),
                    nn.Conv2d(
                        self.INTERNAL[i],
                        self.TARGET[i],
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.TARGET[i]),
                    nn.ReLU6(inplace=True),
                )
            )

        self.encoder_block = nn.ModuleList(enc_blocks)

        self.conv_block_enc = nn.ModuleList([
            DWSepConv(64, 64),
            DWSepConv(128, 128),
            nn.Sequential(
                DWSepConv(256, 256),
                DWSepConv(256, 256)
            ),
            nn.Sequential(
                DWSepConv(512, 512),
                DWSepConv(512, 512)
            ),
            nn.Sequential(
                DWSepConv(512, 512),
                DWSepConv(512, 512)
            )
        ])

        # ------------------------------------------------------------------
        # Efficient Decoder
        # ------------------------------------------------------------------
        self.decoder_block = nn.ModuleList([
            DWSepConv(filter[0], filter[0])
        ])

        for i in range(4):
            self.decoder_block.append(
                DWSepConv(filter[i + 1], filter[i])
            )

        self.conv_block_dec = nn.ModuleList([
            DWSepConv(filter[0], filter[0])
        ])

        for i in range(4):
            if i == 0:
                self.conv_block_dec.append(
                    DWSepConv(filter[i], filter[i])
                )
            else:
                self.conv_block_dec.append(
                    nn.Sequential(
                        DWSepConv(filter[i], filter[i]),
                        DWSepConv(filter[i], filter[i])
                    )
                )

        # ------------------------------------------------------------------
        # Attention modules (unchanged)
        # ------------------------------------------------------------------
        self.encoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])]
        )

        self.decoder_att = nn.ModuleList(
            [nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])]
        )

        self.encoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[1]])]
        )

        self.decoder_block_att = nn.ModuleList(
            [self.conv_layer([filter[0], filter[0]])]
        )

        for j in range(3):
            if j < 2:
                self.encoder_att.append(
                    nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])
                )

                self.decoder_att.append(
                    nn.ModuleList(
                        [self.att_layer([2 * filter[0], filter[0], filter[0]])]
                    )
                )

            for i in range(4):
                self.encoder_att[j].append(
                    self.att_layer(
                        [2 * filter[i + 1], filter[i + 1], filter[i + 1]]
                    )
                )

                self.decoder_att[j].append(
                    self.att_layer(
                        [filter[i + 1] + filter[i], filter[i], filter[i]]
                    )
                )

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 2]])
                )

                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i]])
                )
            else:
                self.encoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )

                self.decoder_block_att.append(
                    self.conv_layer([filter[i + 1], filter[i + 1]])
                )

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        self.down_sampling = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            return_indices=True
        )

        self.up_sampling = nn.MaxUnpool2d(
            kernel_size=2,
            stride=2
        )

        self.logsigma = nn.Parameter(
            torch.FloatTensor([-0.5, -0.5, -0.5])
        )

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

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma
    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    SegNet_MTAN = EfficientSegNet().to(device)
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

    # checkpt_dir = '/kaggle/working/mtan_epoch_100.pth'
    # checkpoint = torch.load(checkpt_dir, map_location=device)

    # SegNet_MTAN.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = 0
    # start_epoch = checkpoint['epoch'] + 1
    # print(f'Model loaded from epoch: {start_epoch}')

    SegNet_MTAN = torch.compile(SegNet_MTAN)

    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN), count_parameters(SegNet_MTAN) / 24981069))
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


