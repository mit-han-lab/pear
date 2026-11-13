from __future__ import annotations

import itertools
import json
from typing import Any

import datasets

from verl.utils.dataset.rl_dataset import RLHFDataset


# data.train_files/val_files are episode files {"env_class": <a family's env class>, "data_source": <env key>, "env":
# {<key>: [values] | "a-b", ...} or a list of such products}: an episode is one combination of a product's values, the
# constructor kwargs of the env class, which the tool config's entry with that env_class plays
class EpisodeDataset(RLHFDataset):
    """verl's dataset over the episodes of its episode files (one family's or several): one row per episode, no prompt,
    no files."""

    def _read_files_and_tokenize(self) -> None:
        if self.max_samples != -1:
            raise ValueError("data.*_max_samples is not supported: size the episode file instead")

        rows = []

        for episode_file_path in self.data_files:
            with open(episode_file_path) as f:
                episode_file = json.load(f)

            for product in episode_file["env"] if isinstance(episode_file["env"], list) else [episode_file["env"]]:
                keys = list(product)
                values = [self._parse_range(v) if isinstance(v, str) else list(v) for v in product.values()]

                for combination in itertools.product(*values):
                    env_args: dict[str, Any] = dict(zip(keys, combination))
                    rows.append(
                        {
                            "data_source": str(env_args[episode_file["data_source"]]),
                            "prompt": [],  # the loop builds the conversation from the simulator's first answer
                            "reward_model": {"style": "rule", "ground_truth": ""},
                            # the row's tool, by verl's per-row tool_selection (EnvironmentTool is named by its env
                            # class), and the env args each run's Episode starts from, as JSON: one struct column would
                            # give a row the other families' keys (None)
                            "extra_info": {
                                "index": len(rows),
                                "tool_selection": [episode_file["env_class"]],
                                "env": json.dumps(env_args),
                            },
                        }
                    )

        self.dataframe = datasets.Dataset.from_list(rows)
        print(f"dataset len: {len(self.dataframe)} episodes from {self.data_files}")

    @staticmethod
    def _parse_range(ranges: str) -> list[int]:
        """Returns the integers of comma-separated numbers and inclusive ranges, as "0-3,7" gives 0, 1, 2, 3, 7."""
        numbers: list[int] = []

        for part in ranges.split(","):
            if "-" in part:
                first, last = part.split("-")
                numbers.extend(range(int(first), int(last) + 1))
            elif part.strip():
                numbers.append(int(part))

        return numbers
