from pose_tracker import read_posetrack_keypoints
from torch.utils.data import DataLoader
from kp_utils import convert_kps
from smooth_bbox import get_all_bbox_params
from img_utils import get_single_image_crop_demo
from demo_utils import smplify_runner
import cv2
import joblib
import numpy as np
import torch

# arguments
IMG_FILE = '/data/edjchen/VIBE/video_data_frames/C0005/frame_0.png' # original image (without VIBE result)
OPENPOSE_JSON_FILE = '/data/edjchen/VIBE/output_openpose_C0005/C0005_000000000000_keypoints.json'
VIBE_OUTPUT_FILE = '/data/edjchen/VIBE/output/C0005_v1/C0005.MP4/vibe_output.pkl'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get openpose tracking results dictionary
posetrack_dict = read_posetrack_keypoints(OPENPOSE_JSON_FILE)
vibe_output = joblib.load(VIBE_OUTPUT_FILE)
img = cv2.cvtColor(cv2.imread(IMG_FILE), cv2.COLOR_BGR2RGB)

joints2d = posetrack_dict[0]['joints2d']
frames = posetrack_dict[0]['frames']

bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
bboxes[:, 2:] = 150. / bboxes[:, 2:]
bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T
joints2d = joints2d[time_pt1:time_pt2]
frames = frames[time_pt1:time_pt2]

j2d = joints2d[0]
bbox = bboxes[0]
norm_img, raw_img, kp_2d = get_single_image_crop_demo(
    img,
    bbox,
    kp_2d=j2d,
    scale=1.0,
    crop_size=224)

nj2d = torch.from_numpy(kp_2d)
norm_joints2d = []
norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))
norm_joints2d = np.concatenate(norm_joints2d, axis=0)
norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

# Obtain pred_pose, pred_betas, pred_cam from VIBE output file
pred_pose = torch.from_numpy(vibe_output[0]['pose'][0,:].reshape(1, -1))
pred_betas = torch.from_numpy(vibe_output[0]['betas'][0,:].reshape(1, -1))
pred_cam = torch.from_numpy(vibe_output[0]['pred_cam'][0,:].reshape(1, -1))

# put tensors on gpu
pred_pose = pred_pose.to(device)
pred_betas = pred_betas.to(device)
pred_cam = pred_cam.to(device)
norm_joints2d = norm_joints2d.to(device)

# Run Temporal SMPLify
update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
    pred_rotmat=pred_pose,
    pred_betas=pred_betas,
    pred_cam=pred_cam,
    j2d=norm_joints2d,
    device=device,
    batch_size=norm_joints2d.shape[0],
    pose2aa=False,
)

breakpoint()


# UPDATE PARAMETERS
# RENDER INTO IMAGE/VIDEO
# SAVE RESULTS





# ========= Save results to a pickle file ========= #
pred_cam = pred_cam.cpu().numpy()
pred_verts = pred_verts.cpu().numpy()
pred_pose = pred_pose.cpu().numpy()
pred_betas = pred_betas.cpu().numpy()
pred_joints3d = pred_joints3d.cpu().numpy()
smpl_joints2d = smpl_joints2d.cpu().numpy()

orig_cam = convert_crop_cam_to_orig_img(
    cam=pred_cam,
    bbox=bboxes,
    img_width=orig_width,
    img_height=orig_height
)

joints2d_img_coord = convert_crop_coords_to_orig_img(
    bbox=bboxes,
    keypoints=smpl_joints2d,
    crop_size=224,
)

output_dict = {
    'pred_cam': pred_cam,
    'orig_cam': orig_cam,
    'verts': pred_verts,
    'pose': pred_pose,
    'betas': pred_betas,
    'joints3d': pred_joints3d,
    'joints2d': joints2d,
    'joints2d_img_coord': joints2d_img_coord,
    'bboxes': bboxes,
    'frame_ids': frames,
}

vibe_results[person_id] = output_dict
