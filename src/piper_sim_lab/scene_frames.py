"""Static scene placement, separate from robot-local kinematics.

Poses are world [x, y, z, qw, qx, qy, qz], metres / wxyz. Objects are
initialized in the table frame, then remain independent free physics bodies.
"""
from dataclasses import dataclass
import numpy as np

IDENTITY_POSE = (0., 0., 0., 1., 0., 0., 0.)


def quat_multiply(a, b, xp=np):
    w,x,y,z = a
    r,s,t,u = b
    return xp.stack([w*r-x*s-y*t-z*u, w*s+x*r+y*u-z*t,
                     w*t-x*u+y*r+z*s, w*u+x*t-y*s+z*r])


@dataclass(frozen=True)
class Pose:
    position: tuple
    quaternion: tuple

    @classmethod
    def parse(cls, value, name):
        a = np.asarray(value, dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError(f'{name} must be seven finite values [x,y,z,qw,qx,qy,qz]')
        norm = np.linalg.norm(a[3:])
        if norm < 1e-12:
            raise ValueError(f'{name} quaternion must be nonzero')
        return cls(tuple(a[:3]), tuple(a[3:]/norm))

    @property
    def rotation(self):
        w,x,y,z = self.quaternion
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    def to_world(self, points, xp=np):
        return xp.asarray(points) @ xp.asarray(self.rotation).T + xp.asarray(self.position)

    def from_world(self, points, xp=np):
        return (xp.asarray(points)-xp.asarray(self.position)) @ xp.asarray(self.rotation)

    def vector_from_world(self, vectors, xp=np):
        return xp.asarray(vectors) @ xp.asarray(self.rotation)

    def quaternion_to_world(self, quaternion, xp=np):
        return quat_multiply(xp.asarray(self.quaternion), quaternion, xp)

    def quaternion_from_world(self, quaternion, xp=np):
        inverse = xp.asarray(self.quaternion)*xp.asarray([1.,-1.,-1.,-1.])
        return quat_multiply(inverse, quaternion, xp)


@dataclass(frozen=True)
class SceneFrames:
    base: Pose
    table: Pose

    @classmethod
    def from_config(cls, config):
        base = config.get('robot_base_pose')
        # Legacy configurations retain their original world-z base translation.
        if base is None:
            base = (0.,0.,config.get('base_height',0.),1.,0.,0.,0.)
        table = config.get('table_surface_pose', IDENTITY_POSE)
        return cls(Pose.parse(base,'robot_base_pose'), Pose.parse(table,'table_surface_pose'))

    def plane_in_base(self, clearance=0.):
        """Unit normal n and offset b (including clearance) for n @ p_base >= b."""
        normal_world = self.table.rotation[:,2]
        normal = self.base.rotation.T @ normal_world
        offset = normal_world @ (np.asarray(self.table.position)-self.base.position)
        return tuple(normal), float(offset+clearance)


def apply_scene_frames(spec, frames):
    # Both supported pinned XMLs have identity fixed ancestors above base_link.
    # Leave the files and the robot-local FK chain unchanged.
    base = spec.body('base_link')
    base.pos = frames.base.position
    base.quat = frames.base.quaternion
    floor = spec.geom('floor')
    floor.pos = frames.table.position
    floor.quat = frames.table.quaternion
