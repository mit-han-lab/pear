The conversation is about to be shortened. This turn writes a summary instead of an action: do not write a json action block.

After this turn, the conversation keeps the system prompt with the task, the task's images and the most recent turns, and your summary replaces everything else. Write what is needed to continue the task from there.

Write the summary inside <summary> and </summary> tags, in these parts:

1. Progress: which parts of the task are done and which are not, judged by what the camera images showed, not by which actions were sent.
2. Actions and results: the actions executed so far, in order and briefly, each with the execution feedback it received; group runs of similar actions. Keep the exact values that later actions depend on, such as positions and poses that worked, targets that were fixed, and where objects were seen.
3. Problems: what failed or did not go as intended, why, and what was learned from it.
4. Next: the immediate next step, and the plan for the rest of the task.

Write only facts from the conversation, and keep the summary short.
