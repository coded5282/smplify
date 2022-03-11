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

import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags
from smpl import get_smpl_faces
import torch

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

def rasterize_mesh(camera, verts, azimuth=0., elev=0., img_dim=224, gt_mask=None,
                    rotation=None, translation=None, focal_length=None, camera_center=None, save_name=None):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        faces = get_smpl_faces()
        faces = faces.astype(np.int32)
        faces = torch.from_numpy(faces)
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
    
        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        mesh = Meshes(
            verts=[verts.to(device)],   
            faces=[faces.to(device)], 
            textures=textures
        )
       
        # Initialize a camera.
        # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
        # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
        R, T = look_at_view_transform(-1, elev, azimuth)

        K = torch.unsqueeze(torch.eye(4), 0).type(torch.float32).to(device)
        K[0, 0, 0] = camera[0]
        K[0, 1, 1] = -camera[1]
        K[0, 0, 3] = camera[2] * camera[0]
        K[0, 1, 3] = -camera[3] * camera[1]
        K[0, 2, 2] = -1 # (1, 4, 4)

        """ batch_size = 1
        K = torch.zeros([batch_size, 3, 3], device=device)
        K[:,0,0] = focal_length
        K[:,1,1] = focal_length
        K[:,2,2] = 1.
        K[:,:-1, -1] = camera_center # (1, 3, 3)
        extra_column = torch.zeros((3, 1)).unsqueeze(0)
        K = torch.cat((K, extra_column), 2)
        extra_row = torch.zeros((1, 4)).unsqueeze(0)
        K = torch.cat((K, extra_row), 1)
        K[0, -1, -1] = 1 """
        """ K[0, 0, -1] = K[0, 0, -2]
        K[0, 1, -1] = K[0, 1, -2]
        K[0, 0, -2] = 0
        K[0, 1, -2] = 0 """

        #return K, R, T
        cameras = FoVPerspectiveCameras(device=device, K=K, R=R, T=T) # R: (1,3,3)   T: (1,3)
        #cameras = FoVPerspectiveCameras(device=device, R=R, T=T) # R: (1,3,3)   T: (1,3)
        #cameras = FoVPerspectiveCameras(device=device, R=rotation, T=translation) # R: (1,3,3)   T: (1,3)
        #cameras = FoVPerspectiveCameras(device=device, K=K, R=rotation, T=translation)
    
        raster_settings = RasterizationSettings(
            image_size=(img_dim, img_dim), 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
    
        raster_out = rasterizer(mesh)
        pix_to_face = raster_out.pix_to_face
        bary_coords = raster_out.bary_coords
        #return pix_to_face[0, ..., 0], bary_coords[0, ..., 0, :]


        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. 
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]]) # -3.0 to put in front of mannequin
        
        # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
        )

        images = renderer(mesh, lights=lights)
        render_image = images[0, ..., :3]
        render_image *= 255.
        b = render_image[:,:,2]
        g = render_image[:,:,1]
        r = render_image[:,:,0]
        """ rgb_image = np.zeros(render_image.shape)
        rgb_image[:,:,0] = b
        rgb_image[:,:,1] = g
        rgb_image[:,:,2] = r """
        gt_mask = gt_mask * 255.
        gt_mask = torch.unsqueeze(gt_mask, -1)
        #gt_mask = torch.mean(gt_mask, 2, True) # flatten to 224x224x1
        g = torch.unsqueeze(g, 2) # add dummy 3rd dimension
        valid_mask = (g != 255)
        gt_mask = gt_mask.to(device)
        output_img = g * valid_mask + (~valid_mask) * gt_mask # 224x224x1 tensor
        
        if save_name is not None:
            cv2.imwrite(save_name, output_img.detach().cpu().numpy())

        render_output = g * valid_mask # to zero out
        render_output[valid_mask] = 1

        return render_output[:,:,0]
        #return valid_mask[:,:,0].int()
        #return pix_to_face[0, ..., 0].detach().cpu().numpy(), bary_coords[0, ..., 0, :].detach().cpu().numpy()

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)
        # breakpoint()

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image

    def render_mask(self, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)
        # breakpoint()

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return valid_mask