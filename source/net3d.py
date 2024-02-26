import torch
import torch.nn as nn 

class CenterOfMassLayer3d(nn.Module):
    def __init__(self):
        super(CenterOfMassLayer3d, self).__init__()

    def forward(self, img):
        """
        x: tensor of shape [n_batch, chs, dim1, dim2, dim3]
        returns: center of mass, shape [n_batch, chs, 3]
        """
        n_batch, chs, _, _, _ = img.shape
        eps = 1e-8

        arange_xyz = [None] * 3
        mass_xyz = [None] * 3
        total_mass_xyz = [None] * 3
        center_mass_xyz = [None] * 3

        for i,(dim,sum_dim) in enumerate(zip([2,3,4], [(2,3),(2,4),(3,4)])):
            arange_xyz[i] = torch.linspace(0,1,img.shape[dim],device=img.device).float().view(1,1,-1).repeat(n_batch, chs, 1)
            mass_xyz[i] = img.sum(dim=sum_dim)
            total_mass_xyz[i] = mass_xyz[i].sum(-1,True) + eps
            center_mass_xyz[i] = (arange_xyz[i]*mass_xyz[i]).sum(-1,True)/total_mass_xyz[i]

        points = torch.cat(center_mass_xyz,-1) #center of mass, shape [n_batch, chs, 2]
        return points

class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3d, self).__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class KeypointDetectorNetwork3d(nn.Module):
    def __init__(self, number_of_keypoints):
        super(KeypointDetectorNetwork3d, self).__init__()

        dims = [1, 32, 64, 64, 128, 128, 256, 256, 512, number_of_keypoints]

        self.net = nn.Sequential(
            ConvBlock3d(dims[0], dims[1]),
            ConvBlock3d(dims[1], dims[2]),
            nn.MaxPool3d(2),
            ConvBlock3d(dims[2], dims[3]),
            ConvBlock3d(dims[3], dims[4]),
            nn.MaxPool3d(2),
            ConvBlock3d(dims[4], dims[5]),
            ConvBlock3d(dims[5], dims[6]),
            nn.MaxPool3d(2),
            ConvBlock3d(dims[6], dims[7]),
            ConvBlock3d(dims[7], dims[8]),
            nn.MaxPool3d(2),
            ConvBlock3d(dims[8], dims[9]),
            CenterOfMassLayer3d()
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
    