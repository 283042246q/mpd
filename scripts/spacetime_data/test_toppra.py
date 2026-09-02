import numpy as np

import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo


DOF = 7


# -------------------------
# 1. 假造一条 7-DoF 几何路径
# -------------------------

s_waypoints = np.linspace(0.0, 1.0, 6)

q_waypoints = np.array(
    [
        [0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.7],
        [0.1, -0.3, 0.1, -1.7, 0.1, 1.5, 0.7],
        [0.2, -0.2, 0.2, -1.6, 0.0, 1.6, 0.8],
        [0.3, -0.1, 0.1, -1.5, -0.1, 1.7, 0.8],
        [0.4, -0.2, 0.0, -1.4, 0.0, 1.8, 0.9],
        [0.5, -0.3, -0.1, -1.3, 0.1, 1.9, 0.9],
    ],
    dtype=np.float64,
)

path = ta.SplineInterpolator(
    s_waypoints,
    q_waypoints,
)


# -------------------------
# 2. 机器人速度/加速度限制
# 注意：下面只是 smoke-test 数值
# 正式生成数据必须换成 Panda 实际采用的 limits
# -------------------------

vmax = np.full(DOF, 1.5)
amax = np.full(DOF, 3.0)

vlim = np.column_stack((-vmax, vmax))
alim = np.column_stack((-amax, amax))


pc_vel = constraint.JointVelocityConstraint(vlim)

pc_acc = constraint.JointAccelerationConstraint(
    alim,
    discretization_scheme=constraint.DiscretizationType.Interpolation,
)


# -------------------------
# 3. TOPP-RA
# -------------------------

instance = algo.TOPPRA(
    [pc_vel, pc_acc],
    path,
    parametrizer="ParametrizeConstAccel",
)


# rest-to-rest:
# s_dot(0) = 0
# s_dot(1) = 0
traj = instance.compute_trajectory(0.0, 0.0)

if traj is None:
    raise RuntimeError("TOPP-RA failed")


T = traj.duration

print("T_min =", T)


# -------------------------
# 4. 检查输出
# -------------------------

t = np.linspace(0.0, T, 500)

q = traj(t)
qd = traj(t, 1)
qdd = traj(t, 2)

print("q shape:", q.shape)
print("max |qd|:", np.max(np.abs(qd), axis=0))
print("max |qdd|:", np.max(np.abs(qdd), axis=0))

print("velocity valid:",
      np.all(np.abs(qd) <= vmax[None, :] + 1e-6))

print("acceleration valid:",
      np.all(np.abs(qdd) <= amax[None, :] + 1e-5))
