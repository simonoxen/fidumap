import matplotlib.pyplot as plt
import torch
import numpy as np

def generate_debug_plt2d(images, keypoints):
    case_idx = torch.randint(images[0].shape[0], (1,))
    images = [img[case_idx,:,:,:].squeeze().cpu().detach().numpy() for img in images]
    keypoints = [kp[case_idx,:,:].squeeze().cpu().detach().numpy() for kp in keypoints]
    fig, axes = plt.subplots(1, len(images))
    for ax,img,kp in zip(axes,images,keypoints):
        ax.imshow(img, cmap='gray', origin='upper', extent=(-1,1,1,-1))
        ax.scatter(kp[:,0], kp[:,1])
    return fig

def generate_debug_plt3d(moving_img,moving_kp,fixed_img,fixed_kp,transformed_img):
    case_idx = torch.randint(moving_img[0].shape[0], (1,))
    kp_idx = torch.randint(moving_kp.shape[1], (1,))
    fig, axes = plt.subplots(3,3)
    generate_img_kp_plot(moving_img, moving_kp, case_idx, kp_idx, axes[0,:])
    generate_img_kp_plot(fixed_img, fixed_kp, case_idx, kp_idx, axes[1,:])
    generate_img_img_plot(fixed_img, transformed_img, case_idx, axes[2,:])
    for ax in axes.flatten():
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig

def generate_img_kp_plot(img_data, kp_coords, b_idx=None, kp_idx=None, axes=None):
    b_idx = b_idx if b_idx is not None else torch.randint(img_data[0].shape[0], (1,))
    kp_idx = kp_idx if kp_idx is not None else torch.randint(kp_coords.shape[1], (1,))
    if axes is None:
        fig, axes = plt.subplots(1,3)
    else:
        fig = None
    other_idx = [(2,0,1), (1,0,2), (0,1,2)]
    r = img_data.shape[-1] / 2
    for i,ax in enumerate(axes):
        slice_idx = torch.round(kp_coords[b_idx,kp_idx,other_idx[i][0]] * r + r).int()
        img_slice = get_img_slice(img_data, b_idx, slice_idx, i)
        ax.imshow(img_slice, cmap='gray', origin='upper', extent=(-1,1,1,-1), aspect='equal')
        ax.scatter(kp_coords[b_idx,kp_idx,other_idx[i][1]].cpu().detach().numpy(),
                   kp_coords[b_idx,kp_idx,other_idx[i][2]].cpu().detach().numpy())
    return fig

def generate_img_img_plot(img_data1, img_data2, b_idx=None, axes=None):
    r = img_data1.shape[-1] / 2
    slice_idx = torch.Tensor([r]).int().to(img_data1.device)
    for i,ax in enumerate(axes):
        stack = []
        for img_data in [img_data1, img_data2]:
            img_slice = get_img_slice(img_data, b_idx, slice_idx, i)
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            stack.append(img_slice)
        stack.append(stack[0])
        ax.imshow(np.stack(stack, axis=-1), origin='upper', extent=(-1,1,1,-1), aspect='equal')

def get_img_slice(img_data, b_idx, slice_idx, axis):
    return torch.index_select(img_data[b_idx,:,:,:,:].squeeze(), axis, slice_idx).squeeze().cpu().detach().numpy()

