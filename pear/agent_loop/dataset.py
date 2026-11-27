from __future__ import annotations

import json

import datasets

from pear.agent_loop.tasks import load_tasks, parse_range
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

        rows = [
            {
                "data_source": task.data_source,
                "prompt": [],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "index": task.index,
                    "tool_selection": [task.env_class],
                    "env": json.dumps(task.env_args),
                },
            }
            for task in load_tasks(self.data_files)
        ]

        self.dataframe = datasets.Dataset.from_list(rows)
        print(f"dataset len: {len(self.dataframe)} episodes from {self.data_files}")

    _parse_range = staticmethod(parse_range)
