import pytest
import koopman_admm as KA
from scenarios import SCENARIOS, run_scenario

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def legs():
    return KA.learn_operators()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_rows_and_steering(name, legs):
    s, _ = run_scenario(name, "rows+steer", legs)
    assert s["upright"] and s["contact_ticks"] == 0 and s["min_clearance"] > 0.0, s
    assert s["end_xy"][0] > max(o[0] for o in SCENARIOS[name]["obst"]) + 0.3, s       # got past them


def test_rows_alone_hold_at_the_wall(legs):
    s, _ = run_scenario("wall", "rows", legs)
    assert s["upright"] and s["contact_ticks"] == 0 and s["min_clearance"] > 0.0 and s["end_xy"][0] < 1.0, s


def test_without_keepout_the_robot_hits_the_cylinder(legs):
    s, _ = run_scenario("headon", "none", legs)
    assert s["contact_ticks"] > 0 and s["min_clearance"] < 0.0, s
