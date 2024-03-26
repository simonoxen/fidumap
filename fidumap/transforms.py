import torch
import torch.nn.functional as F

class AffineTransform:
    def __init__(self, dim, matrix=None, n_batch=1, device=None):
        self.dim = dim
        if matrix is not None:
            assert matrix.shape[1:] == (dim+1, dim+1), "Matrix shape should be (n_batch, %d, %d)" % (dim+1, dim+1)
            self.matrix = matrix
            self.n_batch = matrix.shape[0]
            self.device = matrix.device
        else:
            self.n_batch = n_batch
            self.device = device
            self.matrix = self.batch_eye()

    def batch_eye(self):
        return torch.eye(self.dim+1, device=self.device).reshape((1, self.dim+1, self.dim+1)).repeat(self.n_batch, 1, 1)

    def apply_to_img(self, img):
        """
        img: (n_batch, n_channel, height, width)
        """
        theta = torch.inverse(self.matrix)[:, :self.dim, :]
        affine_grid = F.affine_grid(theta,
                                    img.shape,
                                    align_corners=False)
        
        # take img min to 0 to use 'zeros' padding mode in grid_sample
        img_min = img.min()
        transformed_img = F.grid_sample(img - img_min,
                                        grid=affine_grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False)
        return transformed_img + img_min
    
    def apply_to_kp(self, kp):
        """
        kp: (n_batch, n_kp, dim)
        """
        kp_extend = torch.cat([kp, torch.ones((kp.shape[0],kp.shape[1],1),device=kp.device)], 2)
        transformed_kp_extend = torch.bmm(self.matrix, kp_extend.transpose(1,2)).transpose(1,2)
        transformed_kp = transformed_kp_extend[:,:,:-1]
        return transformed_kp
    
    def save(self, path):
        import SimpleITK as sitk
        transform = sitk.AffineTransform(self.dim)
        transform.SetMatrix([float(x) for x in list(self.matrix[0,0:-1,0:-1].detach().cpu().numpy().flatten())])
        transform.SetTranslation([float(x) for x in list(self.matrix[0,0:-1,-1].detach().cpu().numpy().flatten())])
        sitk.WriteTransform(transform, str(path))

class AffineTransform2D(AffineTransform):
    def __init__(self, matrix=None, n_batch=1, device=None) -> None:
        super().__init__(2, matrix, n_batch, device)

    def randomize_matrix(self, max_scale=0.2, max_translation=0.2, max_rotation=3.1416, max_shear=0.1):
        
        scale       = (torch.rand((self.n_batch, 2), device=self.device) * 2 - 1) * max_scale + 1
        translation = (torch.rand((self.n_batch, 2), device=self.device) * 2 - 1) * max_translation
        rotation    = (torch.rand((self.n_batch),    device=self.device) * 2 - 1) * max_rotation
        shear       = (torch.rand((self.n_batch, 2), device=self.device) * 2 - 1) * max_shear

        scale_transform     = self.batch_eye()
        translate_transform = self.batch_eye()
        rotation_transform  = self.batch_eye()
        shear_transform     = self.batch_eye()

        scale_transform[:, 0, 0] = scale[:, 0]
        scale_transform[:, 1, 1] = scale[:, 1]

        translate_transform[:, :2, 2] = translation

        rotation_transform[:, 0, 0] = torch.cos(rotation)
        rotation_transform[:, 0, 1] = -torch.sin(rotation)
        rotation_transform[:, 1, 0] = torch.sin(rotation)
        rotation_transform[:, 1, 1] = torch.cos(rotation)

        shear_transform[:, 0, 1] = shear[:, 0]
        shear_transform[:, 1, 0] = shear[:, 1]

        composite_transform = torch.bmm(shear_transform, 
                            torch.bmm(scale_transform, 
                            torch.bmm(translate_transform, rotation_transform)))

        self.matrix = composite_transform

class AffineTransform3D(AffineTransform):
    def __init__(self, matrix=None, n_batch=1, device=None) -> None:
        super().__init__(3, matrix, n_batch, device)

    def randomize_matrix(self, max_scale=0.2, max_translation=0.2, max_rotation=3.1416, max_shear=0.1):
        
        scale       = (torch.rand((self.n_batch, 3), device=self.device) * 2 - 1) * max_scale + 1
        translation = (torch.rand((self.n_batch, 3), device=self.device) * 2 - 1) * max_translation
        rotation    = (torch.rand((self.n_batch, 3),    device=self.device) * 2 - 1) * max_rotation
        shear       = (torch.rand((self.n_batch, 6), device=self.device) * 2 - 1) * max_shear

        # aux = translation[0,-2]
        # translation = torch.zeros_like(translation)
        # translation[0,-2] = aux

        scale_transform     = self.batch_eye()
        translate_transform = self.batch_eye()
        rotation_transform_x  = self.batch_eye()
        rotation_transform_y  = self.batch_eye()
        rotation_transform_z  = self.batch_eye()
        shear_transform     = self.batch_eye()

        scale_transform[:, 0, 0] = scale[:, 0]
        scale_transform[:, 1, 1] = scale[:, 1]
        scale_transform[:, 2, 2] = scale[:, 2]

        translate_transform[:, :3, 3] = translation

        rotation_transform_x[:, 1, 1] = torch.cos(rotation[:, 0])
        rotation_transform_x[:, 1, 2] = -torch.sin(rotation[:, 0])
        rotation_transform_x[:, 2, 1] = torch.sin(rotation[:, 0])
        rotation_transform_x[:, 2, 2] = torch.cos(rotation[:, 0])

        rotation_transform_y[:, 0, 0] = torch.cos(rotation[:, 1])
        rotation_transform_y[:, 2, 0] = -torch.sin(rotation[:, 1])
        rotation_transform_y[:, 0, 2] = torch.sin(rotation[:, 1])
        rotation_transform_y[:, 2, 2] = torch.cos(rotation[:, 1])

        rotation_transform_z[:, 0, 0] = torch.cos(rotation[:, 2])
        rotation_transform_z[:, 0, 1] = -torch.sin(rotation[:, 2])
        rotation_transform_z[:, 1, 0] = torch.sin(rotation[:, 2])
        rotation_transform_z[:, 1, 1] = torch.cos(rotation[:, 2])

        shear_transform[:, 0, 1] = shear[:, 0]
        shear_transform[:, 0, 2] = shear[:, 1]
        shear_transform[:, 1, 0] = shear[:, 2]
        shear_transform[:, 1, 2] = shear[:, 3]
        shear_transform[:, 2, 0] = shear[:, 4]
        shear_transform[:, 2, 1] = shear[:, 5]

        rotation_transform = torch.bmm(rotation_transform_z, 
                            torch.bmm(rotation_transform_y, rotation_transform_x))
        
        composite_transform = torch.bmm(shear_transform, 
                            torch.bmm(scale_transform, 
                            torch.bmm(translate_transform, rotation_transform)))

        self.matrix = composite_transform