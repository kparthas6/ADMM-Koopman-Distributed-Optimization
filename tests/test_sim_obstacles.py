"""TorqueSim(obstacles=...): solid geoms added without touching the robot."""
import numpy as np
from torque_sim import TorqueSim


def test_obstacles_are_added_and_robot_unchanged():
    plain = TorqueSim(control_dt=0.005)
    sim = TorqueSim(control_dt=0.005, obstacles=np.array([[1.2, 0.0, 0.15], [2.0, 0.4, 0.2]]))   # ndarray accepted
    assert len(sim.obstacle_gids) == 2 and sim.m.ngeom == plain.m.ngeom + 2
    assert np.isclose(sim.mass, plain.mass)
    assert np.array_equal(sim.qadr, plain.qadr) and np.array_equal(sim.vadr, plain.vadr)
    sim.reset()                                                    # runs collision detection at the home pose
    assert sim.obstacle_contacts() == 0


def test_sphere_and_cylinder_types():
    sim = TorqueSim(control_dt=0.005, obstacles=[(1.0, 0.0, 0.1), (2.0, 0.0, 0.3, 0.2)])
    import mujoco
    types = [sim.m.geom_type[g] for g in sim.obstacle_gids]
    assert types == [mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE]


def test_robot_touching_an_obstacle_is_counted():
    sim = TorqueSim(control_dt=0.005, obstacles=[(0.0, 0.0, 0.1)])   # a pole through the trunk
    sim.reset()
    import mujoco
    mujoco.mj_forward(sim.m, sim.d)
    assert sim.obstacle_contacts() > 0
