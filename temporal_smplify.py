# This script is the extended version of https://github.com/nkolot/SPIN/blob/master/smplify/smplify.py to deal with
# sequences inputs.

import os
import torch
import numpy as np
import colorsys
import trimesh
import math

from smpl import SMPL, JOINT_IDS, SMPL_MODEL_DIR, get_smpl_faces
from losses import temporal_camera_fitting_loss, temporal_body_fitting_loss, geometric_loss
from renderer import Renderer
from renderer import rasterize_mesh

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from prior import MaxMixturePrior

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

def arrange_betas(pose, betas):
    batch_size = pose.shape[0]
    num_video = betas.shape[0]

    video_size = batch_size // num_video
    betas_ext = torch.zeros(batch_size, betas.shape[-1], device=betas.device)
    for i in range(num_video):
        betas_ext[i*video_size:(i+1)*video_size] = betas[i]

    return betas_ext

class TemporalSMPLify():
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 step_size=1e-2,
                 batch_size=66,
                 num_iters=100,
                 focal_length=5000,
                 use_lbfgs=True,
                 device=torch.device('cuda'),
                 max_iter=100):

        # Store options
        self.device = device
        self.focal_length = focal_length
        #self.step_size = step_size
        self.step_size = 1e-1
        self.max_iter = max_iter
        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [JOINT_IDS[i] for i in ign_joints]
        #self.num_iters = num_iters
        self.num_iters = 50

        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder='data/vibe_data',
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        self.use_lbfgs = use_lbfgs
        # Load SMPL model
        self.smpl = SMPL(SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(self.device)

    def __call__(self, 
                init_pose, 
                init_betas, 
                init_cam_t, 
                camera_center, 
                keypoints_2d, 
                gt_mask=None, 
                bboxes=None,
                orig_width=None,
                orig_height=None):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """
        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Initialize renderer for silhouette loss
        if gt_mask is not None:
            renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = False
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True
        body_opt_params = [body_pose, betas, global_orient, camera_translation]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.max_iter,
                                               lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    body_optimizer.zero_grad()
                    betas_ext = arrange_betas(body_pose, betas)
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas_ext)
                    model_joints = smpl_output.joints
                    #breakpoint()

                    # transforming camera matrices and rendering to prepare for silhouette loss
                    mesh_mask = None
                    if gt_mask is not None:
                        model_vertices = smpl_output.vertices
                        #camera_translation_clone = camera_translation.detach().clone()

                        camera_translation_weak = torch.stack([
                                                    2 * 5000. / (224 * camera_translation[:,2] + 1e-9),
                                                    camera_translation[:,0], camera_translation[:,1]
                                                ], dim=-1)

                        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                        rotation = torch.eye(3, device=device).unsqueeze(0).expand(1, -1, -1)
                        sx = camera_translation_weak[:,0]
                        sy = camera_translation_weak[:,0]
                        tx = camera_translation_weak[:,1]
                        ty = camera_translation_weak[:,2]
                        orig_cam = torch.cat([sx, sy, tx, ty]).unsqueeze(0)
                        #orig_cam = torch.hstack((sx, sy, tx, ty)).unsqueeze(0)
                        mesh_mask = rasterize_mesh(orig_cam[0], model_vertices[0], gt_mask=gt_mask, rotation=rotation,
                                                translation=camera_translation_weak, focal_length=5000, camera_center=camera_center)
                        # mesh_mask (224,224) tensor
                        # gt_mask (224,224) tensor

                    model_vertices = smpl_output.vertices

                    """ left_forearm_top_vertex = model_vertices[0,1617,:] # left forearm top
                    left_pinky_vertex = model_vertices[0,2425,:] # left pinky
                    loss_left_arm = geometric_loss(left_forearm_top_vertex, left_pinky_vertex, 0.35)

                    right_forearm_top_vertex = model_vertices[0,5885,:] # right forearm top
                    right_pinky_vertex = model_vertices[0,5086,:] # right pinky
                    loss_right_arm = geometric_loss(right_forearm_top_vertex, right_pinky_vertex, 0.35)

                    arm_loss_sum = (0.01 ** 2) * (loss_left_arm + loss_right_arm) """

                    # model_vertices[0,4721,:] left shoulder (currently .3597)
                    # model_vertices[0,1239,:] right shoulder
                    left_shoulder_vertex = model_vertices[0,4721,:]
                    right_shoulder_vertex = model_vertices[0,1239,:]
                    loss_shoulder = geometric_loss(left_shoulder_vertex, right_shoulder_vertex, 0.20)

                    left_above_hip_vertex = model_vertices[0,4145,:]
                    right_above_hip_vertex = model_vertices[0,655,:]
                    loss_above_hip = geometric_loss(left_above_hip_vertex, right_above_hip_vertex, 0.10)

                    hip_shoulder_loss_sum = (80.0 ** 2) * (loss_shoulder + loss_above_hip)

                    # model_vertices[0,4145,:] left above hip (currently .302m)
                    # model_vertices[0,655,:] right above hip
                    # legs
                    bot_vertex_a = model_vertices[0,3421,:]
                    top_vertex_a = model_vertices[0,809,:]
                    loss_a = geometric_loss(bot_vertex_a, top_vertex_a, 0.975)
                    bot_vertex_b = model_vertices[0,6822,:]
                    top_vertex_b = model_vertices[0,4295,:]
                    loss_b = geometric_loss(bot_vertex_b, top_vertex_b, 0.975)
                    geometric_loss_sum = (80.0 ** 2) * (loss_a + loss_b)

                    loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                             joints_2d, joints_conf, self.pose_prior,
                                             focal_length=self.focal_length, gt_mask=gt_mask, mesh_mask=mesh_mask,
                                             mesh_vertices=model_vertices)

                    loss += geometric_loss_sum
                    loss += hip_shoulder_loss_sum
                    #loss += arm_loss_sum

                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                betas_ext = arrange_betas(body_pose, betas)
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas_ext)
                model_joints = smpl_output.joints
                loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior,
                                         focal_length=self.focal_length)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()
                # scheduler.step(epoch=i)

        # Step 3: Optimize pose only
        # Optimize only the body pose of the body
        camera_translation.requires_grad = False
        body_pose.requires_grad = True
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, global_orient]

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.max_iter,
                                               lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    body_optimizer.zero_grad()
                    betas_ext = arrange_betas(body_pose, betas)
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas_ext)
                    model_joints = smpl_output.joints

                    # transforming camera matrices and rendering to prepare for silhouette loss
                    mesh_mask = None
                    if gt_mask is not None:
                        model_vertices = smpl_output.vertices
                        camera_translation_weak = torch.stack([
                                                    2 * 5000. / (224 * camera_translation[:,2] + 1e-9),
                                                    camera_translation[:,0], camera_translation[:,1]
                                                ], dim=-1)

                        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                        rotation = torch.eye(3, device=device).unsqueeze(0).expand(1, -1, -1)
                        sx = camera_translation_weak[:,0]
                        sy = camera_translation_weak[:,0]
                        tx = camera_translation_weak[:,1]
                        ty = camera_translation_weak[:,2]
                        orig_cam = torch.cat([sx, sy, tx, ty]).unsqueeze(0)
                        #orig_cam = torch.hstack((sx, sy, tx, ty)).unsqueeze(0)
                        mesh_mask = rasterize_mesh(orig_cam[0], model_vertices[0], gt_mask=gt_mask, rotation=rotation,
                                                translation=camera_translation_weak, focal_length=5000, camera_center=camera_center)
                        # mesh_mask (224,224) tensor
                        # gt_mask (224,224) tensor

                    model_vertices = smpl_output.vertices
                    """ left_forearm_top_vertex = model_vertices[0,1617,:] # left forearm top
                    left_pinky_vertex = model_vertices[0,2425,:] # left pinky
                    loss_left_arm = geometric_loss(left_forearm_top_vertex, left_pinky_vertex, 0.35)

                    right_forearm_top_vertex = model_vertices[0,5885,:] # right forearm top
                    right_pinky_vertex = model_vertices[0,5086,:] # right pinky
                    loss_right_arm = geometric_loss(right_forearm_top_vertex, right_pinky_vertex, 0.35)

                    arm_loss_sum = (0.01 ** 2) * (loss_left_arm + loss_right_arm) """
                    # model_vertices[0,4721,:] left shoulder (currently .3597)
                    # model_vertices[0,1239,:] right shoulder
                    left_shoulder_vertex = model_vertices[0,4721,:]
                    right_shoulder_vertex = model_vertices[0,1239,:]
                    loss_shoulder = geometric_loss(left_shoulder_vertex, right_shoulder_vertex, 0.20)

                    left_above_hip_vertex = model_vertices[0,4145,:]
                    right_above_hip_vertex = model_vertices[0,655,:]
                    loss_above_hip = geometric_loss(left_above_hip_vertex, right_above_hip_vertex, 0.10)

                    hip_shoulder_loss_sum = (80.0 ** 2) * (loss_shoulder + loss_above_hip)

                    # model_vertices[0,4145,:] left above hip (currently .302m)
                    # model_vertices[0,655,:] right above hip
                    # legs
                    bot_vertex_a = model_vertices[0,3421,:]
                    top_vertex_a = model_vertices[0,809,:]
                    loss_a = geometric_loss(bot_vertex_a, top_vertex_a, 0.975)
                    bot_vertex_b = model_vertices[0,6822,:]
                    top_vertex_b = model_vertices[0,4295,:]
                    loss_b = geometric_loss(bot_vertex_b, top_vertex_b, 0.975)
                    geometric_loss_sum = (80.0 ** 2) * (loss_a + loss_b)

                    loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                             joints_2d, joints_conf, self.pose_prior,
                                             focal_length=self.focal_length, gt_mask=gt_mask, mesh_mask=mesh_mask,
                                             mesh_vertices=model_vertices)

                    #loss += geometric_loss_sum
                    #loss += hip_shoulder_loss_sum
                    #loss += arm_loss_sum

                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                betas_ext = arrange_betas(body_pose, betas)
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas_ext)
                model_joints = smpl_output.joints
                loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior,
                                         focal_length=self.focal_length)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Step 4: Optimize arms only
        # Optimize only the body arms of the body
        breakpoint()
        camera_translation.requires_grad = False
        body_pose.requires_grad = False
        betas.requires_grad = True
        global_orient.requires_grad = False
        camera_translation.requires_grad = False
        body_opt_params = [betas]

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.max_iter,
                                               lr=self.step_size, line_search_fn='strong_wolfe')
            for i in range(self.num_iters):
                def closure():
                    body_optimizer.zero_grad()
                    betas_ext = arrange_betas(body_pose, betas)
                    smpl_output = self.smpl(global_orient=global_orient,
                                            body_pose=body_pose,
                                            betas=betas_ext)
                    model_joints = smpl_output.joints

                    # transforming camera matrices and rendering to prepare for silhouette loss
                    mesh_mask = None
                    if gt_mask is not None:
                        model_vertices = smpl_output.vertices
                        camera_translation_weak = torch.stack([
                                                    2 * 5000. / (224 * camera_translation[:,2] + 1e-9),
                                                    camera_translation[:,0], camera_translation[:,1]
                                                ], dim=-1)

                        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                        rotation = torch.eye(3, device=device).unsqueeze(0).expand(1, -1, -1)
                        sx = camera_translation_weak[:,0]
                        sy = camera_translation_weak[:,0]
                        tx = camera_translation_weak[:,1]
                        ty = camera_translation_weak[:,2]
                        orig_cam = torch.cat([sx, sy, tx, ty]).unsqueeze(0)
                        #orig_cam = torch.hstack((sx, sy, tx, ty)).unsqueeze(0)
                        mesh_mask = rasterize_mesh(orig_cam[0], model_vertices[0], gt_mask=gt_mask, rotation=rotation,
                                                translation=camera_translation_weak, focal_length=5000, camera_center=camera_center)
                        # mesh_mask (224,224) tensor
                        # gt_mask (224,224) tensor

                    model_vertices = smpl_output.vertices

                    left_forearm_top_vertex = model_vertices[0,1617,:] # left forearm top
                    left_pinky_vertex = model_vertices[0,2425,:] # left pinky
                    loss_left_arm = geometric_loss(left_forearm_top_vertex, left_pinky_vertex, 0.40)

                    right_forearm_top_vertex = model_vertices[0,5885,:] # right forearm top
                    right_pinky_vertex = model_vertices[0,5086,:] # right pinky
                    loss_right_arm = geometric_loss(right_forearm_top_vertex, right_pinky_vertex, 0.40)

                    arm_loss_sum = (10 ** 2) * (loss_left_arm + loss_right_arm)

                    # model_vertices[0,4721,:] left shoulder (currently .3597)
                    # model_vertices[0,1239,:] right shoulder
                    """ left_shoulder_vertex = model_vertices[0,4721,:]
                    right_shoulder_vertex = model_vertices[0,1239,:]
                    loss_shoulder = geometric_loss(left_shoulder_vertex, right_shoulder_vertex, 0.20)

                    left_above_hip_vertex = model_vertices[0,4145,:]
                    right_above_hip_vertex = model_vertices[0,655,:]
                    loss_above_hip = geometric_loss(left_above_hip_vertex, right_above_hip_vertex, 0.10)

                    hip_shoulder_loss_sum = (80.0 ** 2) * (loss_shoulder + loss_above_hip) """

                    # model_vertices[0,4145,:] left above hip (currently .302m)
                    # model_vertices[0,655,:] right above hip
                    # legs
                    """ bot_vertex_a = model_vertices[0,3421,:]
                    top_vertex_a = model_vertices[0,809,:]
                    loss_a = geometric_loss(bot_vertex_a, top_vertex_a, 0.975)
                    bot_vertex_b = model_vertices[0,6822,:]
                    top_vertex_b = model_vertices[0,4295,:]
                    loss_b = geometric_loss(bot_vertex_b, top_vertex_b, 0.975)
                    geometric_loss_sum = (80.0 ** 2) * (loss_a + loss_b)

                    loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                             joints_2d, joints_conf, self.pose_prior,
                                             focal_length=self.focal_length, gt_mask=gt_mask, mesh_mask=mesh_mask,
                                             mesh_vertices=model_vertices) """

                    #loss += geometric_loss_sum
                    #loss += hip_shoulder_loss_sum
                    #loss += arm_loss_sum
                    loss = arm_loss_sum

                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

            for i in range(self.num_iters):
                betas_ext = arrange_betas(body_pose, betas)
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas_ext)
                model_joints = smpl_output.joints
                loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior,
                                         focal_length=self.focal_length)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():
            betas_ext = arrange_betas(body_pose, betas)
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas_ext)
            model_joints = smpl_output.joints

            # transforming camera matrices and rendering to prepare for silhouette loss
            mesh_mask = None
            if gt_mask is not None:
                model_vertices = smpl_output.vertices.detach().clone()
                camera_translation_clone = camera_translation.detach().clone()
                camera_translation_weak = torch.stack([
                                            2 * 5000. / (224 * camera_translation_clone[:,2] + 1e-9),
                                            camera_translation_clone[:,0], camera_translation_clone[:,1]
                                        ], dim=-1)

                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                rotation = torch.eye(3, device=device).unsqueeze(0).expand(1, -1, -1)
                sx = camera_translation_weak[:,0]
                sy = camera_translation_weak[:,0]
                tx = camera_translation_weak[:,1]
                ty = camera_translation_weak[:,2]
                orig_cam = torch.cat([sx, sy, tx, ty]).unsqueeze(0)
                #orig_cam = torch.hstack((sx, sy, tx, ty)).unsqueeze(0)
                mesh_mask = rasterize_mesh(orig_cam[0], model_vertices[0], gt_mask=gt_mask, rotation=rotation,
                                        translation=camera_translation_weak, focal_length=5000, camera_center=camera_center)
                                        
            reprojection_loss = temporal_body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                             joints_2d, joints_conf, self.pose_prior,
                                             focal_length=self.focal_length, gt_mask=gt_mask, mesh_mask=mesh_mask,
                                             mesh_vertices=model_vertices)

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        # Back to weak perspective camera
        camera_translation = torch.stack([
            2 * 5000. / (224 * camera_translation[:,2] + 1e-9),
            camera_translation[:,0], camera_translation[:,1]
        ], dim=-1)

        betas = betas.repeat(pose.shape[0],1)
        output = {
            'theta': torch.cat([camera_translation, pose, betas], dim=1),
            'verts': vertices,
            'kp_3d': joints,
        }

        return output, reprojection_loss
        # return vertices, joints, pose, betas, camera_translation, reprojection_loss

    def get_fitting_loss(self, pose, betas, cam_t, camera_center, keypoints_2d):
        """Given body and camera parameters, compute reprojection loss value.
        Input:
            pose: SMPL pose parameters
            betas: SMPL beta parameters
            cam_t: Camera translation
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            reprojection_loss: Final joint reprojection loss
        """

        batch_size = pose.shape[0]

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        # Split SMPL pose to body pose and global orientation
        body_pose = pose[:, 3:]
        global_orient = pose[:, :3]

        with torch.no_grad():
            smpl_output = self.smpl(global_orient=global_orient,
                                    body_pose=body_pose,
                                    betas=betas, return_full_pose=True)
            model_joints = smpl_output.joints
            reprojection_loss = temporal_body_fitting_loss(body_pose, betas, model_joints, cam_t, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior,
                                                  focal_length=self.focal_length,
                                                  output='reprojection')

        return reprojection_loss
