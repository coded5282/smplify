import cv2
import joblib
import numpy as np
import colorsys
import torch
import os
import joblib
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from pose_tracker import read_posetrack_keypoints
from torch.utils.data import DataLoader
from kp_utils import convert_kps
from smooth_bbox import get_all_bbox_params
from img_utils import get_single_image_crop_demo
from demo_utils import smplify_runner
from demo_utils import convert_crop_cam_to_orig_img
from renderer import Renderer
from smpl import get_smpl_faces

def check_latest(INPUT_DIR):
    current_files = os.listdir(INPUT_DIR)
    try:
        file_nums = [int(f.split('.')[0].split('_')[1]) for f in current_files]
        return sorted(file_nums)[-1]
    except Exception as e:
        print("May have input incorrect directory?")

# arguments
IMG_FILE = '/data/edjchen/VIBE/video_data_frames/man_bool/mannequin_booled.jpg' # original image (without VIBE result)
OPENPOSE_JSON_FILE = '/data/edjchen/VIBE/output_openpose_man_bool/mannequin_booled_000000000000_keypoints.json'
#VIBE_OUTPUT_FILE = '/data/edjchen/VIBE/output/man_bool_v2/mannequin_booled.avi/vibe_output.pkl'
VIBE_OUTPUT_FILE = '/data/edjchen/TSMPlify/pkl_outputs/results_9.pkl'
SILHOUETTE_FILE = '/data/edjchen/VIBE/mannequin_background_zero.png'

MESH_OUTPUT_DIR = '/data/edjchen/TSMPlify/mesh_outputs'
IMG_OUTPUT_DIR = '/data/edjchen/TSMPlify/img_outputs'
PKL_OUTPUT_DIR = '/data/edjchen/TSMPlify/pkl_outputs'

curr_file_num = check_latest(MESH_OUTPUT_DIR) + 1
MESH_OUTPUT_FILE = '/data/edjchen/TSMPlify/mesh_outputs/mesh_' + str(curr_file_num) + '.obj'
IMG_OUTPUT_FILE = '/data/edjchen/TSMPlify/img_outputs/img_' + str(curr_file_num) + '.png'
PKL_OUTPUT_FILE = '/data/edjchen/TSMPlify/pkl_outputs/results_' + str(curr_file_num) + '.pkl'

if not os.path.exists(MESH_OUTPUT_DIR):
    print("Creating mesh output directory")
    os.makedirs(MESH_OUTPUT_DIR)

if not os.path.exists(IMG_OUTPUT_DIR):
    print("Creating image output directory")
    os.makedirs(IMG_OUTPUT_DIR)

if not os.path.exists(PKL_OUTPUT_DIR):
    print("Creating results output directory")
    os.makedirs(PKL_OUTPUT_DIR)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get openpose tracking results dictionary
posetrack_dict = read_posetrack_keypoints(OPENPOSE_JSON_FILE)
vibe_output = joblib.load(VIBE_OUTPUT_FILE)
img = cv2.cvtColor(cv2.imread(IMG_FILE), cv2.COLOR_BGR2RGB)
orig_height, orig_width, orig_channels = img.shape
silhouette_img = cv2.cvtColor(cv2.imread(SILHOUETTE_FILE), cv2.COLOR_BGR2GRAY)
silhouette_mask = silhouette_img != 0
# set silhouette_mask to None to not use sihouette loss

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
try:
    pred_pose = torch.from_numpy(vibe_output[0]['pose'][0,:].reshape(1, -1))
except:
    pred_pose = vibe_output['pose'][0,:].reshape(1, -1)
try:
    pred_betas = torch.from_numpy(vibe_output[0]['betas'][0,:].reshape(1, -1))
except:
    pred_betas = vibe_output['betas'][0,:].reshape(1, -1)
try:
    pred_cam = torch.from_numpy(vibe_output[0]['pred_cam'][0,:].reshape(1, -1))
except:
    pred_cam = vibe_output['pred_cam'][0,:].reshape(1, -1)

""" import math
import trimesh
model_vertices = vibe_output['verts'][0,:]
faces = get_smpl_faces()
mesh = trimesh.Trimesh(vertices=model_vertices.numpy(), faces=faces, process=False)
Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
mesh.apply_transform(Rx)
breakpoint() """

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
    silhouette=silhouette_mask,
    bboxes=bboxes,
    orig_width=orig_width,
    orig_height=orig_height,
)

# Save results dictionary
results_dict = {
    'verts': new_opt_vertices,
    'pred_cam': new_opt_cam,
    'pose': new_opt_pose,
    'betas': new_opt_betas,
    'joints_3d': new_opt_joints3d,
}

joblib.dump(results_dict, PKL_OUTPUT_FILE)
print("Saved results pickle file")

# Render image
renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)

pred_cam = new_opt_cam.numpy()
bboxes = bboxes

orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

img = cv2.imread(IMG_FILE)
frame_verts = new_opt_vertices[0].numpy()
frame_cam = orig_cam[0]
mesh_color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mesh_color,
                    mesh_filename=MESH_OUTPUT_FILE,
                )

cv2.imwrite(IMG_OUTPUT_FILE, img)
print("Saved image file")
cv2.imshow("Render Result", img)
cv2.waitKey(0)