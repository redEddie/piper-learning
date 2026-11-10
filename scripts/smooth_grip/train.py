"""Training entrypoint; the experiment driver starts evaluations afterwards."""
from piper_sim_lab.experiment_guard import require_idle_gpu
require_idle_gpu()
from piper_sim_lab.train import main
main()
