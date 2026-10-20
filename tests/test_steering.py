"""steer_offset: the heading it picks must let the actual body spheres, swept along the walk, clear the obstacles."""
import numpy as np
import pytest
import koopman_admm as KA

POS = np.r_[0.0, 0.0, 0.27]
BOD = np.asarray(KA.KEEPOUT_CFG["bodies"], float)
CLEAR = KA.KEEPOUT_CFG["steer"]["clear"]


def swept_clearance(yaw_body, travel, obs, look=0.8):
    """Smallest clearance of the body spheres translated along `travel` for `look` m (dense sampling)."""
    u, best = np.array([np.cos(travel), np.sin(travel)]), np.inf
    for s in np.linspace(0.0, look, 81):
        for b in BOD:
            p = POS[:2] + s*u + b[0]*np.array([np.cos(yaw_body), np.sin(yaw_body)])
            best = min(best, (np.linalg.norm(p[None] - obs[:, :2], axis=1) - obs[:, 3] - b[3]).min())
    return best


def test_clear_path_keeps_the_command():
    assert KA.steer_offset(POS, 0.0, (0.3, 0.0), KA.obstacle_array([(2.5, 1.5, 0.15)])) == 0.0


def test_no_motion_no_offset():
    assert KA.steer_offset(POS, 0.0, (0.0, 0.0), KA.obstacle_array([(0.5, 0.0, 0.15)])) == 0.0


def test_sphere_above_the_trunk_does_not_block():
    assert KA.steer_offset(POS, 0.0, (0.3, 0.0), KA.obstacle_array([(0.6, 0.0, 1.2, 0.2)])) == 0.0


@pytest.mark.parametrize("cmd", [(0.3, 0.0), (0.3, 0.1), (0.3, 0.2), (0.2, 0.2), (0.0, 0.2)])
def test_chosen_heading_clears_the_obstacle(cmd):
    """The chosen heading clears by `clear`, or, where no heading within reach does, by as much as the best one."""
    a0, offs = np.arctan2(cmd[1], cmd[0]), np.linspace(-1.2, 1.2, 49)
    for ox in np.linspace(0.3, 1.5, 9):
        for oy in np.linspace(-0.6, 0.9, 9):
            obs = KA.obstacle_array([(ox, oy, 0.12)])
            if swept_clearance(0.0, 0.0, obs, look=0.0) < CLEAR:      # already inside the steering clearance
                continue
            off = KA.steer_offset(POS, 0.0, cmd, obs)
            got = swept_clearance(off, a0 + off, obs)
            if got < CLEAR - 0.005:
                best = max(swept_clearance(o, a0 + o, obs) for o in offs)
                assert got > min(CLEAR, best) - 0.005, (cmd, ox, oy, off, got, best)


def test_obstacle_beside_does_not_disable_steering():
    obs = KA.obstacle_array([(0.0, 0.37, 0.15), (0.75, 0.0, 0.15)])  # one beside the trunk, one dead ahead
    off = KA.steer_offset(POS, 0.0, (0.3, 0.0), obs)
    assert off != 0.0 and swept_clearance(off, off, obs[1:]) > 0.0
