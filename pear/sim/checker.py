from __future__ import annotations

import argparse
import json
import pickle
import pkgutil
import time
from pathlib import Path

import yaml

from pear.sim.schemas import ServerContext


class SimulatorChecker:
    """Starts one simulator process of a tool config's stateless env class as a server does and makes one call of an
    episode: its initial observation, or an action when one is given."""

    def __init__(self, args: argparse.Namespace) -> None:
        with open(args.tool_config) as config_file:
            env_class_path = yaml.safe_load(config_file)["tools"][0]["config"]["env_class"]
        self.env_class = pkgutil.resolve_name(env_class_path)
        self.env_args = json.loads(args.env_args)
        self.action = json.loads(args.action)  # None: the initial observation
        self.log_dir = Path(args.log_dir)
        # the tool config of a check: its file's env class and its own log directory, no run's servers
        self.config = {"env_class": env_class_path, "log_dir": args.log_dir}

    @classmethod
    def from_command_line(cls) -> SimulatorChecker:
        """The check of the tool config, the env args and the action the command line names."""
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--tool-config", required=True, help="the family's tool config, which names its env class")
        parser.add_argument("--env-args", required=True, help="the episode's env args, as JSON")
        parser.add_argument("--action", default="null", help="the action to carry out, as JSON (by default none)")
        parser.add_argument("--log-dir", required=True, help="the directory of the process's log")

        return cls(parser.parse_args())

    def run(self) -> None:
        """Starts the simulator process, makes the call and prints what it carried out; exits with the reason of a check
        that failed."""
        # the check in a server's place: its log directory exists as a server's does, it has no GPU of Ray's, and its
        # log is the check's output
        self.log_dir.mkdir(parents=True, exist_ok=True)
        [process], _ = self.env_class.setup_env(ServerContext(self.config, self.log_dir, [], print))

        process.start()
        if not process.wait_ready(self.env_class.START_TIMEOUT_S):
            raise SystemExit(f"the simulator process ended in its start; see {process.log_path}")
        start_time = time.monotonic()
        request = {"env_args": self.env_args, "history": [], "action": self.action}
        result = pickle.loads(process.call(request, self.env_class.ROLLOUT_TIMEOUT_S))
        process.close()

        if "error" in result:
            raise SystemExit(f"{result['error']}see {process.log_path}")
        frame_sizes = [frame.size for frame in result["images"]]
        if len(frame_sizes) != len(self.env_class.CAMERAS):
            raise SystemExit(f"{len(frame_sizes)} frames for the cameras {self.env_class.CAMERAS}")
        if self.action is not None and not result["executed"]:
            raise SystemExit(f"the action {self.action} moved nothing; see {process.log_path}")
        print(
            f"{self.env_class.__name__}: {result['executed']} native actions in {time.monotonic() - start_time:.1f} s, "
            f"frames {frame_sizes}"
        )


if __name__ == "__main__":
    SimulatorChecker.from_command_line().run()
