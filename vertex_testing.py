from pytorch3d.utils import checkerboard, ico_sphere, torus

# mesh_pred.verts_packed().shape[0] x 3  

sphere = ico_sphere(4)
print("sphere", sphere.verts_packed().shape[0])

r = 0.5
R = 1
toruss1 = torus(r, R, 42, 61)
toruss2 = torus(r, R, 61, 42)
print("torus1", toruss1.verts_packed().shape[0])
print("torus2", toruss2.verts_packed().shape[0])

checker1 = checkerboard(25)
print("checker1", checker1.verts_packed().shape[0])

