import numpy as np

HOME_LEG = np.array([0.0, 0.9, -1.8])
LINK = 0.213                                   # thigh = calf length (m)
HIP_OFFSETS = np.array([[0.1881, -0.04675, 0.0], [0.1881, 0.04675, 0.0],
                        [-0.1881, -0.04675, 0.0], [-0.1881, 0.04675, 0.0]])
THIGH_Y = np.array([-0.08, 0.08, -0.08, 0.08])  # thigh lateral mount offset


def fk_leg(q, hip_off, ty):
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)
    p_thigh = np.array([0.0, ty * c1, ty * s1])
    p_calf = np.array([-LINK * s2, LINK * c2 * s1, -LINK * c2 * c1])
    p_foot = np.array([-LINK * s23, LINK * c23 * s1, -LINK * c23 * c1])
    return hip_off + p_thigh + p_calf + p_foot


def jac_leg(q, hip_off, ty, eps=1e-6):
    """Foot Jacobian d(foot)/d(q) (3, 3), by finite differences."""
    J = np.zeros((3, 3))
    f0 = fk_leg(q, hip_off, ty)
    for k in range(3):
        dq = np.zeros(3); dq[k] = eps
        J[:, k] = (fk_leg(q + dq, hip_off, ty) - f0) / eps
    return J
