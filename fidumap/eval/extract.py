import argparse
from pathlib import Path, PurePath

import torch
import torch.nn as nn 
import torchio as tio

from fidumap.net3d import KeypointDetectorNetwork3d

def get_default_model_path(n_keypoints):
    return Path(__file__).parent / f'./data/pretrain_state_{n_keypoints}.h5'

def main(n_keypoints=None, model_load=None, input=None, out_prefix=None):

    input_subjects = [tio.Subject(mri=tio.ScalarImage(input))] if input else None

    dataset_transform = tio.Compose([
            tio.Resample(2),
            tio.CropOrPad(128),
            tio.ToCanonical(),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        ])

    input_set = tio.SubjectsDataset(input_subjects, transform=dataset_transform) if input_subjects else None

    input_loader = torch.utils.data.DataLoader(input_set, batch_size=1, shuffle=False) if input_set else None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device: %s' % device.type)

    model = KeypointDetectorNetwork3d(n_keypoints)
    if not model_load or not Path(model_load).is_file():
        model_load = get_default_model_path(n_keypoints)
    model.load_state_dict(torch.load(model_load, map_location=device))
    model.to(device)

    model.eval()

    for tio_input_img in input_loader:
        input_img_data = tio_input_img['mri'][tio.DATA].float().to(device)
        input_img_affine = tio_input_img['mri'][tio.AFFINE].float().to(device)
        output_fiducials  = model(input_img_data, input_img_affine)
        # save fiducials
        import numpy as np
        np.savetxt(str(PurePath(out_prefix + '_fiducials.csv')), output_fiducials.squeeze(0).cpu().detach().numpy(), delimiter=',')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_keypoints', type=int, required=True)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--out_prefix', type=str, default='')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.n_keypoints, args.model, args.input, args.out_prefix)
