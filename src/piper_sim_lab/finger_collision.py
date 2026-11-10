"""Opt-in SHOW-87 repair derived from the pinned v100 mesh; no asset rewriting."""
import mujoco
import numpy as np


def clipped_triangle_points(triangles, lo, hi):
    """Clip whole triangles against a z slab, including edge-plane intersections."""
    result = []
    for triangle in triangles:
        polygon = list(triangle)
        for z, sign in ((lo, 1), (hi, -1)):
            output = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                ina, inb = sign * (a[2]-z) >= 0, sign * (b[2]-z) >= 0
                if ina:
                    output.append(a)
                if ina != inb:
                    output.append(a + (b-a)*((z-a[2])/(b[2]-a[2])))
            polygon = output
        result.extend(polygon)
    return np.asarray(result)


def add_finger_bridges(spec):
    """Fill the 7.55 mm connector gap without widening either existing pad.

    Four thin slabs bound clipped mesh triangles (not just selected vertices),
    limiting overhang on the taper. Base and fingertip geoms remain unchanged.
    """
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    b6 = model.body('link6').id
    r6, p6 = data.xmat[b6].reshape(3, 3), data.xpos[b6]
    report = {}
    for name in ('link7', 'link8'):
        bid = model.body(name).id
        rb, pb = data.xmat[bid].reshape(3, 3), data.xpos[bid]
        gid = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == bid
                   and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
        mid = model.geom_dataid[gid]
        vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        points = (vertices @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid]-p6) @ r6
        faces = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid]+model.mesh_facenum[mid]]
        def end(part, direction):
            g = model.geom(f'{name}_{part}').id
            center = (data.geom_xpos[g]-p6) @ r6
            half = np.abs(r6.T @ data.geom_xmat[g].reshape(3,3)) @ model.geom_size[g]
            return center[2] + direction*half[2]
        lo, hi = end('base', 1), end('pad', -1)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rb.T@r6).flatten())
        cuts = np.linspace(lo-1e-6, hi+1e-6, 5)
        for i, (z0,z1) in enumerate(zip(cuts[:-1],cuts[1:])):
            samples = clipped_triangle_points(points[faces], z0,z1)
            lower, upper = samples.min(0), samples.max(0)
            box = spec.body(name).add_geom(name=f'{name}_bridge_{i}', type=mujoco.mjtGeom.mjGEOM_BOX)
            box.size = (upper-lower)/2 + 1e-7
            box.pos = rb.T @ (r6 @ ((lower+upper)/2) + p6-pb)
            box.quat = quat
            box.group, box.density = 3, 0
            box.rgba = [.3,.8,.5,.5]
        report[name] = {'gap_mm': float((hi-lo)*1000), 'slabs': 4}
    return report
