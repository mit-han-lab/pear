import os
import pickle
from pathlib import Path

import pytest

from pear.envs.libero.env import LiberoEnv
from pear.sim.schemas import ServerContext

# a simulator process as a LIBERO server starts it (LiberoEnv.setup_env), from the variables of a LIBERO run's nodes
pytestmark = pytest.mark.skipif(
    not {"LIBERO_PYTHON", "LIBERO_MESA_DIR"} <= os.environ.keys(),
    reason="needs the LIBERO simulator environment (LIBERO_PYTHON, LIBERO_MESA_DIR)",
)
SUITE = "libero_10"
TRIAL = 0
TASK_PLATES = 4  # the white mug on the left plate (plate_1), the yellow and white mug on the right plate (plate_2)
TASK_RIGHT_OF = 6  # the chocolate pudding to the right of the plate
TASK_NO_SIDES = 0  # both the alphabet soup and the tomato sauce in the basket


class BodyPositionsEnv(LiberoEnv):
    """LiberoEnv whose results also hold the y position of every body in the trial's initial scene (y_by_body)."""

    def rollout(self, history, action):
        result = super().rollout(history, action)
        self._open_scene()
        sim = self.scene.sim
        result["y_by_body"] = {
            name: float(sim.data.body_xpos[sim.model.body_name2id(name)][1]) for name in sim.model.body_names
        }
        self._close_scene()

        return result


@pytest.fixture(scope="module")
def result_by_task(tmp_path_factory):
    """The initial observations of the tests' tasks, from one simulator process of BodyPositionsEnv."""
    # in a server's place: no GPU of Ray's, the log on the test's output
    [process], _ = BodyPositionsEnv.setup_env(ServerContext({}, tmp_path_factory.mktemp("logs"), [], print))
    # this module, for the process to import BodyPositionsEnv from
    process.variables["PYTHONPATH"] = f"{Path(__file__).parent}{os.pathsep}{process.variables['PYTHONPATH']}"
    process.start()
    assert process.wait_ready(LiberoEnv.START_TIMEOUT_S), (
        f"the simulator process ended in its start; see {process.log_path}"
    )

    result_by_task = {}
    for task_id in (TASK_PLATES, TASK_RIGHT_OF, TASK_NO_SIDES):
        request = {"env_args": {"suite": SUITE, "task_id": task_id, "trial_id": TRIAL}, "history": [], "action": None}
        result_by_task[task_id] = pickle.loads(process.call(request, LiberoEnv.ROLLOUT_TIMEOUT_S))
        assert "error" not in result_by_task[task_id], result_by_task[task_id]["error"]
    process.close()

    return result_by_task


def test_the_left_plate_lies_on_the_side_the_frames_show_as_right(result_by_task):
    y_by_body = result_by_task[TASK_PLATES]["y_by_body"]

    # the prompt: +y points toward the agentview frame's left, so LIBERO's left plate (-y) is on the frame's right; an
    # object's root body is <name>_main
    assert y_by_body["plate_1_main"] < 0 < y_by_body["plate_2_main"]


def test_the_instruction_names_the_sides_as_the_frames_show_them(result_by_task):
    assert result_by_task[TASK_PLATES]["instruction"] == (
        "put the white mug on the right plate and put the yellow and white mug on the left plate"
    )
    assert result_by_task[TASK_RIGHT_OF]["instruction"] == (
        "put the white mug on the plate and put the chocolate pudding to the left of the plate"
    )
    assert result_by_task[TASK_NO_SIDES]["instruction"] == (
        "put both the alphabet soup and the tomato sauce in the basket"
    )
