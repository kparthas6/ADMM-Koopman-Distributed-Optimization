import os, sys, subprocess
import numpy as np
import pytest
import keepout_reference as KR

pytestmark = pytest.mark.slow
HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_trajectories.py")


def trajectories(root, out):
    subprocess.run([sys.executable, "-B", HELPER, root, out], check=True, capture_output=True)
    return np.load(out)


def test_no_obstacle_runs_are_bit_identical(tmp_path):
    explicit = os.environ.get("KEEPOUT_REF")
    ref = KR.extract_reference(str(tmp_path / "ref"), explicit)
    if ref is None:
        if explicit:
            pytest.fail(f"KEEPOUT_REF={explicit} could not be extracted from git")
        pytest.skip("reference commit not available (no git history)")
    old = trajectories(ref, str(tmp_path / "old.npz"))
    new = trajectories(KR.ROOT, str(tmp_path / "new.npz"))
    for name in old.files:
        assert np.array_equal(old[name], new[name]), name
