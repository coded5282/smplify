import cv2
import numpy as np
import os
import joblib
import colorsys
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from renderer import Renderer

IMG_FILE = '/data/edjchen/VIBE/video_data_frames/man_bool/mannequin_booled.jpg' # original image (without VIBE result)
IMG_OUTPUT_FILE = '/data/edjchen/TSMPlify/test_frame.png'
OBJ_FILE = '/data/edjchen/TSMPlify/mesh_outputs/mesh_109.obj'

# Render image
orig_width = 5472
orig_height = 3648
renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)

img = cv2.imread(IMG_FILE)

# read frame verts from obj file
frame_verts = np.zeros((6890, 3))
obj_file = open(OBJ_FILE, 'r')
obj_data = obj_file.readlines()
verts_num = 0
for obj_data_line in obj_data:
    if obj_data_line.strip().split(' ')[0] == 'v':
        line_split = obj_data_line.strip().split(' ')
        x = float(line_split[1])
        y = -1*float(line_split[2])
        z = -1*float(line_split[3])
        frame_verts[verts_num] = np.array([x, y, z])
        verts_num += 1

frame_data = joblib.load('frame_cam.pkl')
frame_cam = frame_data['cam']

mesh_color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)

img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mesh_color,
                    mesh_filename=None,
                )

cv2.imwrite(IMG_OUTPUT_FILE, img)
print("Saved image file")
cv2.imshow("Render Result", img)
cv2.waitKey(0)