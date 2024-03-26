import argparse
from pathlib import Path, PurePath

import torch
import torch.nn as nn 
import torchio as tio

from fidumap.data_manager import get_validation_transform
from fidumap.model import RegistrationModel
from fidumap.net3d import KeypointDetectorNetwork3d
from fidumap.keypoint_aligners import AffineAligner
from fidumap.transforms import AffineTransform3D

def get_default_model_path(n_keypoints):
    return Path(__file__).parent / f'./models/keypoint_detector_{n_keypoints}.h5'

def main(n_keypoints=None, model_load=None, moving=None, fixed=None,  out_prefix=None):

    moving_subjects = [tio.Subject(mri=tio.ScalarImage(moving))] if moving else None
    fixed_subjects = [tio.Subject(mri=tio.ScalarImage(fixed))] if fixed else None

    dataset_transform = tio.Compose([
            tio.Resample(2),
            tio.CropOrPad(128),
            tio.ToCanonical(),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        ])

    moving_set = tio.SubjectsDataset(moving_subjects, transform=dataset_transform) if moving_subjects else None
    fixed_set = tio.SubjectsDataset(fixed_subjects, transform=dataset_transform) if fixed_subjects else None

    moving_loader = torch.utils.data.DataLoader(moving_set, batch_size=1, shuffle=False) if moving_set else None
    fixed_loader = torch.utils.data.DataLoader(fixed_set, batch_size=1, shuffle=False) if fixed_set else None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device: %s' % device.type)

    kpn = KeypointDetectorNetwork3d(n_keypoints)
    if not model_load or not Path(model_load).is_file():
        model_load = get_default_model_path(n_keypoints)
    kpn.load_state_dict(torch.load(model_load, map_location=device))
    
    kpa = AffineAligner()
    
    model = RegistrationModel(kpn, kpa)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    model.eval()

    for tio_moving_img, tio_fixed_img in zip(moving_loader, fixed_loader):
        moving_img = tio_moving_img['mri'][tio.DATA].float().to(device)
        fixed_img = tio_fixed_img['mri'][tio.DATA].float().to(device)
        pred_transform_matrix, moving_kp, fixed_kp  = model(moving_img, None, fixed_img, None)
        transform = AffineTransform3D(pred_transform_matrix)
        transformed_img = transform.apply_to_img(moving_img)
        transform.save(PurePath(out_prefix + '_transform.tfm'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_keypoints', type=int, required=True)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--moving', type=str, default='')
    parser.add_argument('--fixed', type=str, default='')
    parser.add_argument('--out_prefix', type=str, default='')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.n_keypoints, args.model, args.moving, args.fixed, args.out_prefix)
