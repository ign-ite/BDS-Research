import os as _os
# default values — use the project root relative to this file (src/../)
root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
algo = 'c51'
workload = 'jobs.csv'
beta = 0.5
iteration = 10000
fixed_episodic_reward = 10000
epsilon = 0.001
learning_rate = 0.001
gamma = 0.9
# percentage increase of job duration if job type is not matched with placement strategy
placement_penalty = 30
pp_apply = 'true'

# reward settings
use_dirichlet_weights = False

# pricing settings
enable_stochastic_pricing = False
price_variance = 0.05

# scheduler settings
job_batch_size = 1

# agent settings
use_recurrent_policy = False

# workload settings
trace_type = 'synthetic'

# ── Phase-2 Extension flags (safe defaults = Phase-1 behaviour) ────────────

# Extension 1: Multi-objective weight mode
# "fixed_cost"     → w = [0.50, 0.10, 0.10, 0.15, 0.15]
# "fixed_balanced" → w = [0.20, 0.20, 0.20, 0.20, 0.20]  (Phase-1 compatible)
# "auto_dirichlet" → projected-gradient online optimisation
w_mode = "fixed_balanced"

# Extension 2: VM pricing mode injected into env
# "static" → original fixed prices  (Phase-1 compatible)
# "spot"   → Ornstein-Uhlenbeck stochastic pricing
pricing_mode = "static"

# Extension 3: batch size (already controlled by job_batch_size above,
# this alias is used by BatchScheduler when wrapping the env)
batch_B = 1
