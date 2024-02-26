import torch
import torch.nn as nn 

class CenterOfMassLayer2d(nn.Module):
    def __init__(self):
        super(CenterOfMassLayer2d, self).__init__()

    def forward(self, img):
        """
        x: tensor of shape [n_batch, chs, dim1, dim2]
        returns: center of mass, shape [n_batch, chs, 2]
        """
        n_batch, chs, dim1, dim2 = img.shape
        eps = 1e-8
        arangex = torch.linspace(0,1,dim1,device=img.device).float().view(1,1,-1).repeat(n_batch, chs, 1)
        arangey = torch.linspace(0,1,dim2,device=img.device).float().view(1,1,-1).repeat(n_batch, chs, 1)

        mx = img.sum(dim=2) #mass along the dimN, shape [n_batch, chs, dimN]
        Mx = mx.sum(-1, True) + eps #total mass along dimN

        my = img.sum(dim=3)
        My = my.sum(-1, True) + eps

        cx = (arangex*mx).sum(-1,True)/Mx #center of mass along dimN, shape [n_batch, chs, 1]
        cy = (arangey*my).sum(-1,True)/My

        points = torch.cat([cx,cy],-1) #center of mass, shape [n_batch, chs, 2]
        return points

class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2d, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class KeypointDetectorNetwork2d(nn.Module):
    def __init__(self, number_of_keypoints):
        super(KeypointDetectorNetwork2d, self).__init__()

        dims = [1, 32, 64, 64, 128, 128, 256, 256, 512, number_of_keypoints]

        self.net = nn.Sequential(
            ConvBlock2d(dims[0], dims[1]),
            ConvBlock2d(dims[1], dims[2]),
            nn.MaxPool2d(2),
            ConvBlock2d(dims[2], dims[3]),
            ConvBlock2d(dims[3], dims[4]),
            nn.MaxPool2d(2),
            ConvBlock2d(dims[4], dims[5]),
            ConvBlock2d(dims[5], dims[6]),
            nn.MaxPool2d(2),
            ConvBlock2d(dims[6], dims[7]),
            ConvBlock2d(dims[7], dims[8]),
            nn.MaxPool2d(2),
            ConvBlock2d(dims[8], dims[9]),
            CenterOfMassLayer2d()
        )

    def forward(self, data, affine=None):
        points_normalized = self.net(data)
        if points_normalized.isnan().any():
            raise RuntimeError("NaNs in output.")
        # if affine is None:
        points = points_normalized * 2 - 1 # for grid sample

        # else:
        #     n_batch, chs, dim1, dim2, dim3 = data.shape
        #     normalized_to_ijk = torch.eye(3).reshape((1, 3, 3)).repeat(n_batch, 1, 1).to(data.device)
        #     for i,dim in enumerate([dim1, dim2, dim3]):
        #         normalized_to_ijk[:,i,i] = dim
        #     points_ijk = torch.bmm(points_normalized, normalized_to_ijk)
        #     points_ijk = torch.cat((points_ijk, torch.ones(n_batch,chs,1)), -1)
        #     points_world = torch.bmm(points_ijk, affine.reshape((1, 4, 4)).repeat(n_batch, 1, 1))
        #     points = points_world[:,:,:3] # in world coords
        return points
