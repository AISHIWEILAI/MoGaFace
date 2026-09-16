import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
from scene.backbone.Bisenet import BiSeNet
import torch.nn.functional as F
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, con_input):
        return self.conv(con_input)

class Encoder(nn.Module):
    def __init__(self, in_ch):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_0 = DoubleConv(64, 128)
        self.pool2_0 = nn.MaxPool2d(2)
        self.conv3_0 = DoubleConv(128, 256)
        self.pool3_0 = nn.MaxPool2d(2)
        self.conv4_0 = DoubleConv(256, 512)
        self.pool4_0 = nn.MaxPool2d(2)
        self.conv5_0 = DoubleConv(512, 1024)
        self.pool5_0 = nn.MaxPool2d(2)
        self.conv6_0 = DoubleConv(1024, 2048)
        self.pool6_0 = nn.MaxPool2d(8)

        d_model = 2048
        nhead = 8
        dropout = 0.1
        self.attention_layer = nn.MultiheadAttention(embed_dim=d_model,
                                                     num_heads=nhead,
                                                     batch_first=True,
                                                     dropout=dropout).cuda()

    def forward(self, x0):
        c1_0 = self.conv1(x0)
        c1_0 = self.bn1(c1_0)
        c1_0 = self.relu(c1_0)
        p1_0 = self.maxpool(c1_0)

        c2_0 = self.conv2_0(p1_0)
        p2_0 = self.pool2_0(c2_0)
        c3_0 = self.conv3_0(p2_0)
        p3_0 = self.pool3_0(c3_0)
        c4_0 = self.conv4_0(p3_0)
        p4_0 = self.pool4_0(c4_0)
        c5_0 = self.conv5_0(p4_0)
        p5_0 = self.pool5_0(c5_0)
        c6_0 = self.conv6_0(p5_0)
        p6_0 = self.pool6_0(c6_0).reshape(1, x0.shape[0], -1)

        attn_output, attn_output_weights = self.attention_layer(p6_0, p6_0, p6_0)
        x = attn_output.squeeze(0).mean(dim=0, keepdim=True)
        return x

class MGCGModule(nn.Module):
    def __init__(self, device='cuda'):
        super(MGCGModule, self).__init__()
        self.device = device
        self.mgcg_encoder = Encoder(in_ch=3).to(self.device)

        self.expression_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 100)
        )

        self.transform_norm = transforms.Compose([
            transforms.CenterCrop((512, 512)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])]
        )

        self.mask_model = BiSeNet(n_classes=10)
        # face_mask.pth only initializes BiSeNet; val inference never uses mask_model
        # and checkpoint load_state_dict overwrites mask_model weights anyway.
        self.mask_model.eval()

    def get_face_mask(self, img_norm):
        img = img_norm  # 5,3, 512,512
        out = self.mask_model(img)[0]
        parsing = out.detach().cpu().numpy().argmax(1)  # 5, 512, 512
        face_mask = parsing.copy().astype(np.int32)

        face_mask[(face_mask == 8) | (face_mask == 9)] = 1
        face_mask[(face_mask == 3) | (face_mask == 2)] = 1
        face_mask[(face_mask == 6)] = 1

        parse_dilate = parsing.copy().astype(np.uint8)

        parse_dilate[(parse_dilate == 8) | (parse_dilate == 9)] = 0
        parse_dilate[(parse_dilate == 3) | (parse_dilate == 2)] = 0
        parse_dilate[(parse_dilate == 6)] = 0

        bg_index = (parse_dilate == 0)
        parse_face = parse_dilate.copy()
        parse_face[parse_face > 1] = 1
        parse_dilate[parse_dilate == 1] = 0
        kernel_2 = np.ones((10, 10), dtype=np.uint8)  # kernel size
        parse_dilate = cv2.dilate(parse_dilate, kernel_2, 1)
        parse_dilate = parse_dilate + parse_face
        parse_dilate[bg_index] = 0
        parse_dilate = parse_dilate.astype(np.float32)
        parse_dilate[parse_dilate == 1] = 0.50
        weight_map = parse_dilate.copy().astype(np.float32)
        return face_mask, weight_map

    def process(self, img, current_img_idx=None):
        img_norm = self.transform_norm(img)     # 5, 3, 512, 512
        assert img_norm.shape[2] == 512
        assert img_norm.shape[3] == 512

        if current_img_idx is not None:
            face_mask, weight_map = self.get_face_mask(img_norm)
            face_mask = torch.from_numpy(face_mask).unsqueeze(1).to(self.device)
            weight_map = torch.from_numpy(weight_map).unsqueeze(1).to(self.device)
        else:
            face_mask = None
            weight_map = None

        return img_norm.to(self.device), face_mask, weight_map

    def forward(self, images, current_img_idx=None):
        img_norm, face_mask, weight_map = self.process(images, current_img_idx)      # weight_map.shape 1,1,512,512

        imag_cat = img_norm
        if face_mask is not None and weight_map is not None:
            if images.shape[2] != weight_map.shape[2] or images.shape[3] != weight_map.shape[3]:
                pad_height = images.shape[2] - weight_map.shape[2]
                pad_width = images.shape[3] - weight_map.shape[3]

                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left

                weight_map = F.pad(weight_map, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)
                face_mask = F.pad(face_mask, (pad_left, pad_right, pad_top, pad_bottom), "constant", 1)

        features = self.mgcg_encoder(imag_cat)     # 1, 2048
        flame_out = self.expression_head(features)      # 1,100/118

        flame_params = {
            'expr': torch.tanh(flame_out[:, :100])
        }

        return flame_params, face_mask, weight_map, features
