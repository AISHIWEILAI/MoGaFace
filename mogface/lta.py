import torch
import torch.nn as nn
import torch.nn.functional as F

from scene.encoding import get_encoder

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x)

        return x

class MLP_wight(nn.Module):
    def __init__(self, dims, last_op=None):
        super(MLP_wight, self).__init__()

        self.dims = dims
        self.skip_layer = [int(len(dims) / 2)]
        self.last_op = last_op

        self.layers = []
        for l in range(0, len(dims) - 1):
            if l in self.skip_layer:
                self.layers.append(nn.Conv1d(dims[l] + dims[0], dims[l + 1], 1))
            else:
                self.layers.append(nn.Conv1d(dims[l], dims[l + 1], 1))
            self.add_module("conv%d" % l, self.layers[l])

    def forward(self, latet_code, return_all=False):
        y = latet_code
        tmpy = latet_code
        y_list = []
        for l, f in enumerate(self.layers):
            if l in self.skip_layer:
                y = self._modules['conv' + str(l)](torch.cat([y, tmpy], 1))
            else:
                y = self._modules['conv' + str(l)](y)
            if l != len(self.layers) - 1:
                y = F.leaky_relu(y)
        if self.last_op:
            y = self.last_op(y)
            y_list.append(y)
        if return_all:
            return y_list
        else:
            return y

class LatentTextureAttention(nn.Module):
    def __init__(self,):
        super(LatentTextureAttention, self).__init__()

        self.bound = 0.15
        # image network
        self.lta_image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lta_image_fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.02, inplace=True),
        )

        # attn network
        self.lta_attention = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv1d(128, 3 + 1 + 4 + 3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02, inplace=True)
        )

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=36, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 5
        self.hidden_dim = 64

        self.out_dim = 10144
        self.sigma_net = MLP(self.in_dim, self.out_dim, self.hidden_dim, self.num_layers)
        self.xyz_weight = MLP(3, 1, 16, 3)

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz

    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)

        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    def forward(self, x, image_feats):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)  # 5143, 60

        h = self.sigma_net(enc_x)   # 5143, 11044

        enc_image = image_feats  # [1, 2048]
        enc_image = self.lta_image_fc(enc_image)              # [1, 512]
        enc_w = enc_image.repeat(enc_x.shape[0], 1)  # [5143, 512]

        weight = self.lta_attention(enc_w.unsqueeze(0).permute(0, 2, 1)).squeeze(0)

        d_xyz = torch.matmul(weight.clone(), h).permute(1, 0)
        return d_xyz

if __name__ == '__main__':
    import torch.optim as optim
    model = LatentTextureAttention().cuda()
    x = torch.randn((5143, 3)).cuda()
    image = torch.randn((1, 2048)).cuda()
    y = model(x, image)
    criterion = nn.MSELoss()

    loss = criterion(y+1, y)

    loss.backward()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    optimizer.step()

    print("Loss:", loss.item())
