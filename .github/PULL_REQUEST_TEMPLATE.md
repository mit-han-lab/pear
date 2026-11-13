### What does this PR do?

<!-- One or two sentences. Link the issues and PRs it relates to. -->

### How was it verified?

<!-- The evidence for each claim: the tests added or run, and for what CI cannot run (GPU, simulators, a training run) the
     command, the hardware and the result. A change meant to keep behaviour says how that was shown. -->

### Checklist

- [ ] The title reads `[module] type: description`, for example `[envs] fix: the reset of a stale episode`. CI checks it
      against the modules and types of `.github/scripts/pr_title.py`.
- [ ] One coherent change; unrelated changes go in their own PR.
- [ ] `pre-commit run --all-files` passes.
- [ ] Tests cover the change, or the section above says why they cannot.
- [ ] A change to a prompt file shows the rendered prompt before and after.
- [ ] A change to `3rdparty/*/patches` was exported by the scripts there.
- [ ] No cluster path, host name, account name or credential in the code, the comments or this description.
