"""
==============================================
Visualize Pressure Field of Two Moving Objects
==============================================
"""
print(__doc__)

import numpy as np
import pytransform3d.visualizer as pv
from distance3d import visualization, pressure_field


class AnimationCallback:
    def __init__(self, n_frames, rigid_body1, rigid_body2, position_offset):
        self.n_frames = n_frames
        self.rigid_body1 = rigid_body1
        self.rigid_body2 = rigid_body2
        self.position_offset = position_offset
        self.rigid_body1.express_in(self.rigid_body2.body2origin_)
        self.mesh1 = visualization.TetraMesh(
            self.rigid_body1.body2origin_, self.rigid_body1.vertices_,
            self.rigid_body1.tetrahedra_)
        self.mesh2 = visualization.TetraMesh(
            self.rigid_body2.body2origin_, self.rigid_body2.vertices_,
            self.rigid_body2.tetrahedra_)
        contact_surface = pressure_field.find_contact_surface(
            self.rigid_body1, self.rigid_body2)
        self.contact_surface = visualization.ContactSurface(
            contact_surface.frame2world,
            contact_surface.contact_polygons,
            contact_surface.contact_polygon_triangles,
            contact_surface.pressures)
        self.rigid_body1.body2origin_[:3, 3] += self.position_offset

    def add_artists(self, fig):
        self.mesh1.add_artist(fig)
        self.mesh2.add_artist(fig)
        self.contact_surface.add_artist(fig)

    def __call__(self, step):
        # TODO clean up move interface, introduce RigidBodyArtist?
        if step == 0:
            self.rigid_body1.body2origin_[:3, 3] += -self.position_offset
            self.mesh1.set_data(self.rigid_body2.body2origin_)
        self.rigid_body1.body2origin_[:3, 3] += self.position_offset / self.n_frames
        mesh12origin = np.copy(self.mesh1.mesh2origin)
        mesh12origin[:3, 3] += self.position_offset / self.n_frames
        self.mesh1.set_data(mesh12origin)

        contact_surface = pressure_field.find_contact_surface(
            self.rigid_body1, self.rigid_body2)
        self.contact_surface.set_data(
            contact_surface.frame2world,
            contact_surface.contact_polygons,
            contact_surface.contact_polygon_triangles,
            contact_surface.pressures)
        return self.mesh1, self.contact_surface


cube12origin = np.eye(4)
rigid_body1 = pressure_field.RigidBody.make_cube(cube12origin, 0.1)
cube22origin = np.eye(4)
cube22origin[:3, 3] = np.array([0.0, 0.05, 0.08])
rigid_body2 = pressure_field.RigidBody.make_cube(cube22origin, 0.1)

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)

n_frames = 100
animation_callback = AnimationCallback(
    n_frames, rigid_body1, rigid_body2, np.array([0.1, 0.0, 0.0]))
animation_callback.add_artists(fig)
fig.view_init()
if "__file__" in globals():
    fig.animate(animation_callback, n_frames, loop=True, fargs=())
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
