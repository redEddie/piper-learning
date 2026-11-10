"""Native IK and collision-filtered, genuinely held low placing reset poses."""
import numpy as np
import mujoco


def build_near_table(env, seeds):
    c=env._config;m=env._mj_model;d=mujoco.MjData(m)
    lower,upper=np.asarray(env._lower),np.asarray(env._upper)
    jacp,jacr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
    fingers={m.body('link7').id,m.body('link8').id}
    robot={m.body(f'link{i}').id for i in range(1,9)}
    table=[];rejected=dict(ik=0,collision=0,grasp=0)
    # Existing solved placing poses supply reachable XY/yaw targets. Preserve
    # the actual gripper/cube geometry; never make a hand pass through the bowl.
    for seed in np.asarray(seeds)[::max(len(seeds)//32,1)]:
        d.qpos[:]=np.asarray(env._initial_qpos);d.qpos[env._qpos_ids]=seed
        mujoco.mj_kinematics(m,d)
        target=d.site_xpos[env._tcp_id].copy()
        goal=d.site_xmat[env._tcp_id].reshape(3,3).copy()
        for h in np.linspace(c.gentle_near_min,c.gentle_near_max,3):
            target[2]=c.bowl_floor+c.block_half+h
            q=seed.astype(float).copy()
            for _ in range(180):
                d.qpos[:]=np.asarray(env._initial_qpos);d.qpos[env._qpos_ids]=q
                mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
                rot=d.site_xmat[env._tcp_id].reshape(3,3)
                ep=target-d.site_xpos[env._tcp_id]
                er=.5*sum(np.cross(rot[:,i],goal[:,i]) for i in range(3))
                if np.linalg.norm(ep)<.0001 and np.linalg.norm(er)<.001:break
                mujoco.mj_jacSite(m,d,jacp,jacr,env._tcp_id)
                jac=np.concatenate([jacp,jacr])[:,env._qvel_ids]
                q=np.clip(q+.5*jac.T@np.linalg.solve(jac@jac.T+1e-4*np.eye(6),np.r_[ep,er]),lower,upper)
            if np.linalg.norm(ep)>.001 or np.linalg.norm(er)>.01:
                rejected['ik']+=1;continue
            d.qpos[env._qpos_ids]=q;mujoco.mj_kinematics(m,d)
            tcp=d.site_xpos[env._tcp_id].copy()
            axis=d.site_xmat[env._tcp_id].reshape(3,3)[:,env._closing_axis]
            yaw=np.arctan2(axis[1],axis[0])
            d.qpos[env._finger_qpos_ids]=[c.block_half,-c.block_half]
            d.qpos[env._block_qpos_adr:env._block_qpos_adr+7]=np.r_[tcp,np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
            d.qpos[env._bowl_qpos_adr:env._bowl_qpos_adr+7]=np.r_[tcp[:2],0.,1.,0.,0.,0.]
            d.ctrl[:]=np.asarray(env._initial_ctrl);d.ctrl[env._arm_ctrl_ids]=q;d.ctrl[env._gripper_ctrl_id]=0.
            mujoco.mj_forward(m,d)
            touches=set();invalid=False
            for j in range(d.ncon):
                con=d.contact[j]
                if con.dist>.0001:continue
                a,b=int(m.geom_bodyid[con.geom1]),int(m.geom_bodyid[con.geom2])
                if (a in robot and b==env._bowl_id) or (b in robot and a==env._bowl_id):invalid=True
                if (a in robot and con.geom2==env._floor_geom_id) or (b in robot and con.geom1==env._floor_geom_id):invalid=True
                if a in fingers and b==env._block_id:touches.add(a)
                if b in fingers and a==env._block_id:touches.add(b)
            if invalid:rejected['collision']+=1;continue
            if touches!=fingers:rejected['grasp']+=1;continue
            table.append(q.astype(np.float32))
    env.near_table_report=dict(accepted=len(table),rejected=rejected,requested_height_range_m=[c.gentle_near_min,c.gentle_near_max])
    if not table:raise ValueError(f'No collision-free low held resets: {env.near_table_report}')
    return np.stack(table)
