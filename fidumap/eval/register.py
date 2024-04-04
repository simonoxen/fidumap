import argparse
from pathlib import Path, PurePath

import torch
import torch.nn as nn 
import torchio as tio

from fidumap.model import RegistrationModel
from fidumap.net3d import KeypointDetectorNetwork3d
from fidumap.keypoint_aligners import AffineAligner
from fidumap.transforms import AffineTransform3D

def get_default_model_path(n_keypoints):
    return Path(__file__).parent / f'./data/train_state_{n_keypoints}.h5'

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
    model.to(device)

    model.eval()

    for tio_moving_img, tio_fixed_img in zip(moving_loader, fixed_loader):
        # load data
        moving_img_data = tio_moving_img['mri'][tio.DATA].float().to(device)
        moving_img_affine = tio_moving_img['mri'][tio.AFFINE].float().to(device)
        fixed_img_data = tio_fixed_img['mri'][tio.DATA].float().to(device)
        fixed_img_affine = tio_fixed_img['mri'][tio.AFFINE].float().to(device)
        # eval
        pred_transform_matrix, moving_kp, fixed_kp  = model(moving_img_data, moving_img_affine, fixed_img_data, fixed_img_affine)
        # save transform
        transform = AffineTransform3D(pred_transform_matrix)
        transform.save(PurePath(out_prefix + '_transform.tfm'))
        # save fiducials
        import numpy as np
        np.savetxt(str(PurePath(out_prefix + '_fiducials_moving.csv')), moving_kp.squeeze(0).cpu().detach().numpy(), delimiter=',')
        np.savetxt(str(PurePath(out_prefix + '_fiducials_fixed.csv')), fixed_kp.squeeze(0).cpu().detach().numpy(), delimiter=',')
        # save img
        import SimpleITK as sitk
        sitk_mov_img = sitk.ReadImage(moving)
        sitk_ref_img = sitk.ReadImage(fixed)
        sitk_transform = sitk.ReadTransform(str(PurePath(out_prefix + '_transform.tfm')))
        filter = sitk.ResampleImageFilter()
        filter.SetReferenceImage(sitk_ref_img)
        filter.SetTransform(sitk_transform)
        sitk_out_image = filter.Execute(sitk_mov_img)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(str(PurePath(out_prefix + '_img.nrrd')))
        writer.Execute(sitk_out_image)

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
