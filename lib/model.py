import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNeRF(nn.Module):
    def __init__(self, D: int = 6, W: int = 128):
        super(TinyNeRF, self).__init__()

        self.linear1 = nn.Linear(3 + 3 * 2 * D, W)
        self.linear2 = nn.Linear(W, W)
        self.linear3 = nn.Linear(W, 4)
        self.relu = F.relu

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class NeRF(nn.Module):
    def __init__(
            self,
            D: int = 8,
            W: int = 256,
            skip: int = 4,
            encoding_xyz: int = 6,
            encoding_dir: int = 4):
        super(NeRF, self).__init__()

        self.dim_xyz = 3 + 2 * 3 * encoding_xyz
        self.dim_dir = 3 + 2 * 3 * encoding_dir

        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz, 256))

        for i in range(1, 8):
            if i == 4:
                self.layers_xyz.append(
                    torch.nn.Linear(
                        self.dim_xyz + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))

        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))

        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz:]

        for i in range(8):
            if i == 4:
                x = self.layers_xyz[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_xyz[i](x)

            x = self.relu(x)

        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)

        if self.use_viewdirs:
            x = self.layers_dir[0](torch.cat((feat, dirs), -1))

        else:
            x = self.layers_dir[0](feat)

        x = self.relu(x)

        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)

        rgb = self.fc_rgb(x)

        return torch.cat((rgb, alpha), dim=-1)
