import importlib.util

# The normal trainer environment does not install optional service backends. Their CPU CI job installs the extra.
collect_ignore_glob = []
if importlib.util.find_spec("tinker") is None or importlib.util.find_spec("tinker_cookbook") is None:
    collect_ignore_glob = ["test_tinker*.py"]
