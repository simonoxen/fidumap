import torch
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter

import torchio as tio

from pathlib import Path, PurePath
import argparse

from fidumap.data_manager import get_train_val_sets
from fidumap.net3d import KeypointDetectorNetwork3d
from fidumap.model import RegistrationModel
from fidumap.keypoint_aligners import AffineAligner
from fidumap.transforms import AffineTransform3D
from fidumap.utils import generate_debug_plt3d

def validation_loop(moving_dataloader, fixed_dataloader, model, loss_fn, device):

    running_loss = 0.0
    
    with torch.no_grad():
        for tio_moving_img, tio_fixed_img in zip(moving_dataloader,fixed_dataloader):
            fixed_img = tio_fixed_img['mri'][tio.DATA].float().to(device)
            moving_img = tio_moving_img['mri'][tio.DATA].float().to(device)
            pred_transform_matrix, _, _  = model(moving_img, None, fixed_img, None)
            transformed_img = AffineTransform3D(pred_transform_matrix).apply_to_img(moving_img)
            running_loss += loss_fn(transformed_img, fixed_img).item()
    
    return running_loss / len(moving_dataloader)

def train_loop(moving_dataloader, fixed_dataloader, model, loss_fn, optimizer, device, args):

    running_loss = 0.0
    
    for tio_moving_img, tio_fixed_img in zip(moving_dataloader,fixed_dataloader):

        fixed_img = tio_fixed_img['mri'][tio.DATA].float().to(device)
        moving_img = tio_moving_img['mri'][tio.DATA].float().to(device)

        pred_transform_matrix, moving_kp, fixed_kp = model(moving_img, None, fixed_img, None)
        pred_transform = AffineTransform3D(pred_transform_matrix)

        transformed_img = pred_transform.apply_to_img(moving_img)

        loss = loss_fn(transformed_img, fixed_img)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    debug_plt = generate_debug_plt3d(moving_img, moving_kp, fixed_img, fixed_kp, transformed_img)
    loss_value = running_loss / len(moving_dataloader)

    return loss_value, debug_plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_keypoints', type=int, required=True)
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--same_moving_fixed', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()

def main():
    args = parse_args()

    data_path = Path('/mnt/arbeit/simon/repo/fidumap/data3d/images-resampled')
    image_paths = sorted(data_path.glob('*.nii'))
    training_set, validation_set = get_train_val_sets(image_paths)
    batch_size = 1
    shuffle = False if args.same_moving_fixed else True

    training_moving_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    training_fixed_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    validation_moving_dataloader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    validation_fixed_dataloader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device: %s' % device.type)

    kpn = KeypointDetectorNetwork3d(args.n_keypoints)
    if Path(args.load).is_file():
        kpn.load_state_dict(torch.load(args.load))
    
    kpa = AffineAligner()
    
    model = RegistrationModel(kpn, kpa)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    writer = SummaryWriter(comment=PurePath(args.save).stem)

    best_val_loss = float("Inf")
    
    for t in range(args.n_epochs):
        
        print("Epoch %d/%d\t" % (t+1, args.n_epochs), end="", flush=True)

        model.train()
        train_loss, debug_plt = train_loop(training_moving_loader, training_fixed_loader, model, loss_fn, optimizer, device, args)
        
        model.eval()
        val_loss = validation_loop(validation_moving_dataloader, validation_fixed_dataloader, model, loss_fn, device)

        print("Loss: train %.5f; validation %.5f" % (train_loss, val_loss))

        writer.add_scalar('Loss/train', train_loss, t+1)
        writer.add_scalar('Loss/validation', val_loss, t+1)
        writer.add_figure('moving_fixed_transformed', debug_plt, t+1)

        compare_val = train_loss if args.same_moving_fixed else val_loss
        if args.save and (compare_val < best_val_loss):
            best_val_loss = compare_val
            torch.save(kpn.state_dict(), args.save)

if __name__ == "__main__":
    main()