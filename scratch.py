from smpl import get_smpl_faces
import trimesh

faces = get_smpl_faces()
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
mesh.apply_transform(Rx)

mesh.export(mesh_filename)
mesh.vertices