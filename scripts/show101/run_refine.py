"""Start the authorized placing refinement; caller supplies the preserved checkpoint."""
from piper_sim_lab.experiment_guard import require_idle_gpu
require_idle_gpu()
from piper_sim_lab.train import main
main()
