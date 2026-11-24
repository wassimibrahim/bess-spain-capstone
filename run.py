import sys; sys.path.insert(0, "src")
from bess.pipeline import run_all
run_all(use_cache=("--fresh" not in sys.argv))
