# Recipes

A recipe is one shell script. Its settings are the variables at the top of the script. A training or evaluation
recipe runs on the head of a Ray cluster, and its extra arguments are Hydra overrides.

- **[models/qwen3_8_27b_pear](models/qwen3_8_27b_pear)**: Qwen3.8-27B's recipes
  - **[train](models/qwen3_8_27b_pear/train/README.md)**: Robot-use RL in three stages, on LIBERO and RoboDojo
  - **[deploy](models/qwen3_8_27b_pear/deploy/README.md)**: Served by vLLM on one GB300 or RTX 5090, and the robot
    station that runs with it
- **[eval](eval/README.md)**: One evaluation per benchmark: LIBERO, LIBERO-Plus, LIBERO-Pro, RoboDojo, RoboLab-120
  and nine general benchmarks
