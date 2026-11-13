import time

from pear.monitor.driver_log import DriverLog

BATCH_GROUPS = 129  # the groups a step trains on
FILLER_BYTES = 1_200_000  # the log a rollout writes after its gen line: three times the first window
T0 = time.mktime((2026, 10, 4, 0, 0, 0, 0, 0, -1))  # an earlier launch's last lines
LAUNCHED = T0 + 1800  # this launch's tracker starts
T1 = T0 + 3600  # this launch's step begins


class FakeDriverLog:
    """A synthetic driver log in train.log's format: the driver's stamped lines (phases, sampler polls) and filler."""

    def __init__(self, path) -> None:
        self.path = path
        self.lines: list[str] = []

    def phase(self, at: float, name: str) -> "FakeDriverLog":
        return self.stamped(at, f"phase {name}")

    def poll(self, at: float, running: int, finished: int) -> "FakeDriverLog":
        return self.stamped(at, f"pending: 0, running: {running}, finished: {finished}, failure: 0")

    def filler(self, at: float) -> "FakeDriverLog":
        size = 0
        while size < FILLER_BYTES:
            self.stamped(at, "episode finished " + "x" * 100)
            size += len(self.lines[-1])
        return self

    def stamped(self, at: float, body: str) -> "FakeDriverLog":
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(at))
        self.lines.append(f"\x1b[36m(TaskRunnerV1 pid=1, ip=192.0.2.1)\x1b[0m INFO:{stamp},529:{body}")
        return self

    def write(self) -> str:
        self.path.write_text("\n".join(self.lines) + "\n")
        return str(self.path)


def test_the_gen_phase_is_read_from_a_line_far_back_in_the_log(tmp_path):
    log = FakeDriverLog(tmp_path / "train.log").phase(T1, "step").phase(T1, "gen").filler(T1 + 60)

    state = DriverLog.read(log.poll(T1 + 2400, 600, 90).write(), BATCH_GROUPS, LAUNCHED)

    assert (state.phase, state.phase_start) == ("gen", T1)
    assert (state.available_group_n, state.inflight_group_n) == (90, 600)
    assert not state.is_training


def test_the_training_started_at_the_first_phase_after_gen(tmp_path):
    log = FakeDriverLog(tmp_path / "train.log").phase(T1, "step").phase(T1, "gen").filler(T1 + 60)
    log.poll(T1 + 2340, 600, 140).phase(T1 + 2400, "old_log_prob").filler(T1 + 2460)

    state = DriverLog.read(log.phase(T1 + 2700, "update_actor").write(), BATCH_GROUPS, LAUNCHED)

    assert (state.phase, state.phase_start, state.training_start) == ("update_actor", T1 + 2700, T1 + 2400)
    assert state.available_group_n == 140 - BATCH_GROUPS  # the trainer took its batch after the poll


def test_the_lines_of_an_earlier_launch_are_left_out(tmp_path):
    # a cold start: the earlier launch's gen line and its last poll, then this launch's lines without a phase yet
    log = FakeDriverLog(tmp_path / "train.log").phase(T0, "gen").poll(T0 + 60, 500, 70).filler(LAUNCHED + 60)

    state = DriverLog.read(log.write(), BATCH_GROUPS, LAUNCHED)

    fields = (state.phase, state.phase_start, state.available_group_n, state.inflight_group_n)
    assert fields == (None, None, None, None)


def test_this_launch_s_step_is_read_past_the_lines_of_an_earlier_launch(tmp_path):
    log = FakeDriverLog(tmp_path / "train.log").phase(T0, "update_actor").poll(T0 + 60, 500, 70).filler(LAUNCHED + 60)
    log.phase(T1, "step").phase(T1, "gen").poll(T1 + 60, 600, 10)

    state = DriverLog.read(log.write(), BATCH_GROUPS, LAUNCHED)

    assert (state.phase, state.phase_start, state.available_group_n) == ("gen", T1, 10)
