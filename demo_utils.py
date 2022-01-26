# temporal smplify runner

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import cv2
import time
import json
import torch
import subprocess
import numpy as np
import os.path as osp
#from pytube import YouTube
from collections import OrderedDict

from geometry import rotation_matrix_to_angle_axis
from temporal_smplify import TemporalSMPLify

def smplify_runner(
        pred_rotmat,
        pred_betas,
        pred_cam,
        j2d,
        device,
        batch_size,
        lr=1.0,
        opt_steps=1,
        use_lbfgs=True,
        pose2aa=True,
        silhouette=None,
        bboxes=None,
        orig_width=None,
        orig_height=None,
):
    smplify = TemporalSMPLify(
        step_size=lr,
        batch_size=batch_size,
        num_iters=opt_steps,
        focal_length=5000.,
        use_lbfgs=use_lbfgs,
        device=device,
        # max_iter=10,
    )
    # Convert predicted rotation matrices to axis-angle
    if pose2aa:
        pred_pose = rotation_matrix_to_angle_axis(pred_rotmat.detach()).reshape(batch_size, -1)
    else:
        pred_pose = pred_rotmat

    # Calculate camera parameters for smplify
    pred_cam_t = torch.stack([
        pred_cam[:, 1], pred_cam[:, 2],
        2 * 5000 / (224 * pred_cam[:, 0] + 1e-9)
    ], dim=-1)

    gt_keypoints_2d_orig = j2d
    # Before running compute reprojection error of the network
    opt_joint_loss = smplify.get_fitting_loss(
        pred_pose.detach(), pred_betas.detach(),
        pred_cam_t.detach(),
        0.5 * 224 * torch.ones(batch_size, 2, device=device),
        gt_keypoints_2d_orig).mean(dim=-1)

    best_prediction_id = torch.argmin(opt_joint_loss).item()
    pred_betas = pred_betas[best_prediction_id].unsqueeze(0)
    # pred_betas = pred_betas[best_prediction_id:best_prediction_id+2] # .unsqueeze(0)
    # top5_best_idxs = torch.topk(opt_joint_loss, 5, largest=False)[1]
    # breakpoint()

    start = time.time()
    # Run SMPLify optimization initialized from the network prediction
    # new_opt_vertices, new_opt_joints, \
    # new_opt_pose, new_opt_betas, \
    # new_opt_cam_t, \
    output, new_opt_joint_loss = smplify(
        pred_pose.detach(), pred_betas.detach(),
        pred_cam_t.detach(),
        0.5 * 224 * torch.ones(batch_size, 2, device=device),
        gt_keypoints_2d_orig,
        gt_mask=silhouette,
        bboxes=bboxes,
        orig_width=orig_width,
        orig_height=orig_height,
    )
    new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)
    # smplify_time = time.time() - start
    # print(f'Smplify time: {smplify_time}')
    # Will update the dictionary for the examples where the new loss is less than the current one
    update = (new_opt_joint_loss < opt_joint_loss)

    new_opt_vertices = output['verts']
    new_opt_cam_t = output['theta'][:,:3]
    new_opt_pose = output['theta'][:,3:75]
    new_opt_betas = output['theta'][:,75:]
    new_opt_joints3d = output['kp_3d']

    return_val = [
        update, new_opt_vertices.cpu(), new_opt_cam_t.cpu(),
        new_opt_pose.cpu(), new_opt_betas.cpu(), new_opt_joints3d.cpu(),
        new_opt_joint_loss, opt_joint_loss,
    ]

    return return_val


def trim_videos(filename, start_time, end_time, output_filename):
    command = ['ffmpeg',
               '-i', '"%s"' % filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_filename]
    # command = ' '.join(command)
    subprocess.call(command)


def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder


def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)


def download_ckpt(outdir='data/vibe_data', use_3dpw=False):
    os.makedirs(outdir, exist_ok=True)

    if use_3dpw:
        ckpt_file = 'data/vibe_data/vibe_model_w_3dpw.pth.tar'
        url = 'https://www.dropbox.com/s/41ozgqorcp095ja/vibe_model_w_3dpw.pth.tar'
        if not os.path.isfile(ckpt_file):
            download_url(url=url, outdir=outdir)
    else:
        ckpt_file = 'data/vibe_data/vibe_model_wo_3dpw.pth.tar'
        url = 'https://www.dropbox.com/s/amj2p8bmf6g56k6/vibe_model_wo_3dpw.pth.tar'
        if not os.path.isfile(ckpt_file):
            download_url(url=url, outdir=outdir)

    return ckpt_file


def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam

          
def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    # import IPython; IPython.embed(); exit()
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]

    # unnormalize to crop coords
    keypoints = 0.5 * crop_size * (keypoints + 1.0)

    # rescale to orig img crop
    keypoints *= h[..., None, None] / crop_size

    # transform into original image coords
    keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
    keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
    return keypoints

          
def prepare_rendering_results(vibe_results, nframes):
    frame_results = [{} for _ in range(nframes)]
    for person_id, person_data in vibe_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            frame_results[frame_id][person_id] = {
                'verts': person_data['verts'][idx],
                'cam': person_data['orig_cam'][idx],
            }

    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in enumerate(frame_results):
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )

    return frame_results