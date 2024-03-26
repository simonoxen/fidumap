import torch
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter

import torchio as tio
import numpy as np

from pathlib import Path, PurePath
import argparse

from fidumap.data_manager import get_pretrain_set
from fidumap.net3d import KeypointDetectorNetwork3d
from fidumap.transforms import AffineTransform3D
from fidumap.utils import generate_img_kp_plot

def labels_to_kps(labels_data):
    n_batch = labels_data.shape[0]
    kp_coords = torch.zeros((n_batch, 32, 3), dtype=torch.float32, device=labels_data.device)

    for i,kp_label in enumerate(range(1,33)):
        b,c,z,y,x= torch.where(labels_data == kp_label)
        kp_coords[b, i, 0] = x.float()
        kp_coords[b, i, 1] = y.float()
        kp_coords[b, i, 2] = z.float()

    r = labels_data.shape[-1] / 2
    kp_coords = (kp_coords - r) / r

    return kp_coords

def train_loop(fixed_dataloader, model, loss_fn, optimizer, device, args):

    running_loss = 0.0
    
    for tio_fixed_img in fixed_dataloader:

        fixed_img = tio_fixed_img['mri'][tio.DATA].float().to(device)
        labels_data = tio_fixed_img['labels'][tio.DATA].float().to(device)

        fixed_kp = labels_to_kps(labels_data)

        aux_transform = AffineTransform3D(n_batch=fixed_img.shape[0], device=device)
        aux_transform.randomize_matrix()

        true_moving_kp = aux_transform.apply_to_kp(fixed_kp)

        moving_img = aux_transform.apply_to_img(fixed_img)

        pred_moving_kp = model(moving_img)

        loss = loss_fn(pred_moving_kp, true_moving_kp)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    debug_plt = generate_img_kp_plot(moving_img, pred_moving_kp)
    loss_value = running_loss / len(fixed_dataloader)

    return loss_value, debug_plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_keypoints', type=int, required=True)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--n_epochs', type=int, default=2000)
    return parser.parse_args()

def main():
    args = parse_args()

    data_path = Path('/mnt/arbeit/simon/repo/fidumap/data3d/labels-resampled')
    image_paths = sorted(data_path.glob('*.nii'))
    training_set = get_pretrain_set(image_paths)
    batch_size = 1

    training_fixed_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running on device: %s' % device.type)

    model = KeypointDetectorNetwork3d(args.n_keypoints)
    if Path(args.load).is_file():
        model.load_state_dict(torch.load(args.load))
    
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.to(device)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    writer = SummaryWriter(comment=PurePath(args.save).stem)
    
    for t in range(args.n_epochs):
        
        print("Epoch %d/%d\t" % (t+1, args.n_epochs), end="", flush=True)

        model.train()
        train_loss, debug_plt = train_loop(training_fixed_loader, model, loss_fn, optimizer, device, args)
        
        print("Loss: train %.5f" % train_loss)

        writer.add_scalar('Loss/train', train_loss, t+1)
        writer.add_figure('moving_fixed_transformed', debug_plt, t+1)

    if args.save:
        torch.save(model.state_dict(), args.save)

if __name__ == "__main__":
    main()