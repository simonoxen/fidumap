import torch.nn as nn

class RegistrationModel(nn.Module):
    def __init__(self, keypoint_detector_net, keypoint_aligner):
        super(RegistrationModel, self).__init__()
        self.keypoint_detector_net = keypoint_detector_net
        self.keypoint_aligner = keypoint_aligner

    def forward(self, moving_img_data, moving_img_affine, fixed_img_data, fixed_img_affine):
        moving_kp = self.keypoint_detector_net(moving_img_data, moving_img_affine)
        fixed_kp = self.keypoint_detector_net(fixed_img_data, fixed_img_affine)
        transform = self.keypoint_aligner.alignKeypoints(moving_kp, fixed_kp)
        return transform, moving_kp, fixed_kp