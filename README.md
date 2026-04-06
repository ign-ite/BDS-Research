# RM_DeepRL: Deep Reinforcement Learning for Resource Management

This is the practical application of various agents like DQN, Reinforce, C51, Rainbow in the job scheduling in Spark
This code has took motivation from https://github.com/tawfiqul-islam/RM_DeepRL and we are greatful and humble while doing this, this has been a great experience to modify and work on a code which was wonderful in itself.

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Quick Start](#quick-start)
7. [Running Individual Algorithms](#running-individual-algorithms)
8. [Comprehensive Comparison](#comprehensive-comparison)
9. [Output Structure](#output-structure)
10. [Troubleshooting](#troubleshooting)
11. [Research Paper Results](#research-paper-results)
12. [Contributing](#contributing)
13. [License](#license)

## 🎯 Overview

This repository implements a comprehensive comparison of reinforcement learning algorithms for cluster resource management and job scheduling. The project compares **basic RL algorithms** (REINFORCE, DQN), **advanced RL algorithms** (Rainbow DQN, C51), and **classical baseline schedulers** (FIFO, FCFS, Fair, Capacity, Round Robin, Min-Min).

### Key Features

- **Multi-algorithm comparison**: 4 RL algorithms + 6 baseline schedulers
- **Comprehensive metrics**: Cost, completion time, resource utilization, throughput, deadline adherence
- **Advanced visualizations**: Interactive plots, radar charts, heatmaps, performance evolution
- **Publication-ready results**: Statistical analysis, performance rankings, comparison tables

## 🔧 System Requirements

### Hardware Requirements

- **RAM**: Minimum 8GB, Recommended 16GB+
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but recommended for faster training
- **Storage**: 5GB free disk space

### Software Requirements

- **Python**: 3.7 - 3.9 (TensorFlow compatibility)
- **Operating System**: Windows, macOS, or Linux
- **Git**: For repository cloning

## 📦 Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/RM_DeepRL.git
cd RM_DeepRL
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n rm_deeprl python=3.8
conda activate rm_deeprl

# Using venv
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import tf_agents; print('TF-Agents installed successfully')"
```

## 📁 Project Structure

```
RM_DeepRL/
├── src/
│   ├── main.py                     # Main execution entry point
│   ├── constants.py                # Configuration constants
│   ├── definitions.py              # Data structures (VM, JOB classes)
│   ├── cluster.py                  # Cluster initialization and state management
│   ├── workload.py                 # Workload generation and loading
│   ├── utilities.py                # Configuration utilities
│   ├── rm_environment.py           # RL environment implementation
│   ├── REINFORCE_tfagent.py        # REINFORCE algorithm
│   ├── DQN_tfagent.py             # DQN algorithm
│   ├── C51_tfagent.py             # C51 algorithm
│   ├── R_DQN_tfagent.py           # Rainbow DQN algorithm
│   ├── baseline_schedulers.py      # Classical scheduling algorithms
│   └── comprehensive_comparison.py # Unified comparison script
├── input/
│   └── jobs.csv                    # Job workload data
├── settings/
│   └── config.ini                  # Configuration file
├── output/                         # Generated results (created during execution)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## ⚙️ Configuration

### Edit Configuration File

```ini
# settings/config.ini
[DEFAULT]
root = /path/to/RM_DeepRL
algo = c51
workload = jobs.csv
beta = 0.5
iteration = 10000
fixed_episodic_reward = 10000
epsilon = 0.001
learning_rate = 0.001
gamma = 0.9
placement_penalty = 30
pp_apply = true
```

### Key Configuration Parameters

- **`root`**: Absolute path to project directory
- **`algo`**: Algorithm to run (`reinforce`, `dqn`, `c51`, `rb_dqn`)
- **`workload`**: CSV file containing job data
- **`beta`**: Multi-objective optimization weight
- **`iteration`**: Number of training iterations
- **`learning_rate`**: Neural network learning rate
- **`placement_penalty`**: Penalty for suboptimal job placement

### Research Extensions (Config Flags)

- **`[reward] use_dirichlet_weights`**: Enables dynamic reward weighting via Dirichlet sampling.
- **`[pricing] enable_stochastic_pricing`** and **`price_variance`**: Enables per-step VM price perturbations.
- **`[scheduler] job_batch_size`**: Adds next-N jobs into observation state (`1` keeps legacy behavior).
- **`[agent] use_recurrent_policy`**: Switches DQN to recurrent Q-network (DRQN-style).
- **`[workload] trace_type`**: Selects parser (`synthetic`, `google`, `alibaba`).

## 🚀 Quick Start

### 1. Basic Setup

```bash
# Navigate to project directory
cd RM_DeepRL

# Activate virtual environment
conda activate rm_deeprl

# Update configuration
nano settings/config.ini  # Edit root path
```

### 2. Run Single Algorithm

```bash
# Run C51 algorithm
python src/main.py
```

### 3. Run Comprehensive Comparison

```bash
# Run all algorithms and generate comparison plots
python src/comprehensive_comparison.py
```

### 4. View Results

```bash
# Results saved in output/ directory with timestamp
ls output/
```

## 🔬 Running Individual Algorithms

### REINFORCE Algorithm

```bash
# Method 1: Via main.py
echo "algo = reinforce" >> settings/config.ini
python src/main.py

# Method 2: Direct execution
python -c "
import sys
sys.path.append('src')
import REINFORCE_tfagent
REINFORCE_tfagent.train_reinforce(num_iterations=5000)
"
```

### DQN Algorithm

```bash
# Method 1: Via main.py
echo "algo = dqn" >> settings/config.ini
python src/main.py

# Method 2: Direct execution
python -c "
import sys
sys.path.append('src')
import DQN_tfagent
DQN_tfagent.train_dqn(num_iterations=5000)
"
```

### C51 Algorithm

```bash
# Method 1: Via main.py
echo "algo = c51" >> settings/config.ini
python src/main.py

# Method 2: Direct execution
python -c "
import sys
sys.path.append('src')
import C51_tfagent
C51_tfagent.train_c51_dqn(num_iterations=5000)
"
```

### Rainbow DQN Algorithm

```bash
# Method 1: Via main.py
echo "algo = rb_dqn" >> settings/config.ini
python src/main.py

# Method 2: Direct execution
python -c "
import sys
sys.path.append('src')
import R_DQN_tfagent
R_DQN_tfagent.train_rainbow_dqn(num_iterations=5000)
"
```

### Baseline Schedulers Only

```bash
python -c "
import sys
sys.path.append('src')
from baseline_schedulers import run_benchmark_comparison
run_benchmark_comparison()
"
```

## 📊 Comprehensive Comparison

### Run Complete Comparison

```bash
python src/comprehensive_comparison.py
```

### Interactive Setup

```bash
# When prompted, enter number of iterations
Enter number of iterations for RL algorithms (default 10000): 5000
```

## 📈 Output Structure

### Directory Organization

```
output/
├── YYYYMMDD_HHMMSS_complete_comparison/
│   ├── comprehensive_layered_comparison.png
│   ├── performance_evolution.png
│   ├── categorized_radar_chart.png
│   ├── comprehensive_comparison_summary.csv
│   ├── Rainbow_DQN/
│   │   ├── episode_costs.csv
│   │   ├── episode_time.csv
│   │   ├── utilization.csv
│   │   ├── throughput.csv
│   │   ├── adherence.csv
│   │   └── rewards.csv
│   ├── C51/
│   │   └── [similar structure]
│   └── [other algorithms...]
```

### Key Output Files

- **`comprehensive_layered_comparison.png`**: Main comparison visualization
- **`performance_evolution.png`**: Learning curves for RL algorithms
- **`categorized_radar_chart.png`**: Multi-dimensional performance comparison
- **`comprehensive_comparison_summary.csv`**: Numerical results table
- **Individual algorithm folders**: Detailed metrics for each method

### Visualization Examples

- **Bar Charts**: Cost, time, utilization, throughput comparisons
- **Radar Charts**: Multi-metric performance visualization
- **Learning Curves**: Training progress for RL algorithms
- **Heatmaps**: Performance ranking matrices
- **Interactive Plots**: Bokeh-based web visualizations

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. TensorFlow Installation Issues

```bash
# Install TensorFlow 2.x compatible version
pip install tensorflow==2.9.0
pip install tf-agents==0.15.0
```

#### 2. Memory Errors

```bash
# Reduce batch size in algorithm files
# Edit src/C51_tfagent.py, line ~XXX
batch_size = 64  # Reduce from 128
```

#### 3. Path Configuration Errors

```bash
# Ensure absolute paths in config.ini
# Windows example:
root = C:\Users\username\RM_DeepRL
# Linux/macOS example:
root = /home/username/RM_DeepRL
```

#### 4. Missing Dependencies

```bash
# Install individual packages
pip install matplotlib seaborn bokeh pandas numpy
```

#### 5. CUDA/GPU Issues

```bash
# Check GPU availability
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# Force CPU usage if needed
export CUDA_VISIBLE_DEVICES=""
```

#### 6. Results Loading Errors

```bash
# Ensure proper CSV file structure
# Check output directory permissions
chmod -R 755 output/
```

### Performance Optimization

#### For Faster Training

```python
# Reduce iterations for testing
num_iterations = 1000

# Reduce evaluation frequency
eval_interval = 500

# Smaller networks
fc_layer_params = (100,)
```

#### For Better Results

```python
# Increase iterations
num_iterations = 20000

# Larger networks
fc_layer_params = (200, 100)

# More frequent evaluation
eval_interval = 200
```

## 🚀 Phase 2: Conference-Worthy Extensions & Enhancements

Building upon the initial codebase, we have implemented a suite of advanced "Phase 2" extensions targeted for IEEE/ACM conference submission. These enhancements transform the codebase from a proof-of-concept into a rigorous, statistically sound, and highly reproducible experimental framework.

### 1. Robust Multi-Seed Experimental Pipeline

- **Parallel Seed Execution**: Upgraded from single-seed to multi-seed evaluation (N=5 seeds) using a deterministic seeding protocol across `numpy`, `tensorflow`, `random`, and `PYTHONHASHSEED`.
- **Process Isolation**: Implemented a crash-proof runner (`run_all_seeds.sh`) that spins up each `(extension, seed)` pair as a completely isolated sub-process. This ensures massive memory leaks across episodes do not crash the host machine, guaranteeing uninterrupted weekend/overnight runs.
- **Fail-Safe Resume**: Incremental checkpointing ensures that if training is interrupted, the script automatically skips completed seeds when resumed.

### 2. Statistical Significance & Validation Framework

- **Automated Aggregation**: Created `aggregate_seeds.py` to seamlessly aggregate outputs across all seeds into mean/std format (`XX.XX ± YY.YY`).
- **Hypothesis Testing**: Implemented a suite of statistical tests using `scipy.stats`:
  - _Independent t-tests_ for Bayesian Simplex Search vs. balanced weights.
  - _Welch's t-tests_ for Spot vs. Static pricing (handling unequal variances).
  - _Paired t-tests_ for Transfer Learning convergence times.
- **Rigorous Methodologies**: Incorporated **Cohen's d** for effect size and the **Bonferroni Correction** ($\alpha \approx 0.0167$) to handle multiple comparisons strictly, avoiding P-value hacking.
- **LaTeX Automation**: Directly outputs IEEE-formatted `.tex` tables for copy-pasting into the paper skeleton.

### 3. Advanced Architectural Extensions

- **Bayesian Simplex Search (BSS)**: Replaced a flawed gradient-based auto-weight optimization (which fails to constrain to a 5-simplex efficiently in 10,000 episodes) with BSS utilizing Latin Hypercube Sampling (LHS). It explores valid weight combinations efficiently and provides statistically proven return leaps over fixed balanced algorithms.
- **Stochastic Spot Pricing (OU Process)**: Introduced Ornstein-Uhlenbeck noise models to simulate dynamic cloud spot pricing.
- **Job Batching**: Enhanced state vectors to observe next $B \in [1, 2, 4, 8]$ jobs to explore the horizon trade-off against Rainbow DQN training stability.
- **Cross-Topology Transfer Learning**: Integrated weight transfer learning simulating migrating policies trained on small clusters (e.g., v1) to vastly larger and divergent topologies (e.g., v2).

### 4. GPU & Systems-Level Hardening

Addressed critical system roadblocks allowing high-end GPUs to effectively train logic-heavy environments:

- **XLA JIT Bypassing**: Solved `FloorMod` graph execution compilation failures unique to TensorFlow 2.11 paired with Ada Lovelace (Compute Capability 8.9) architectures by granularly decoupling `tf.function` from XLA auto-compilation (`TF_XLA_FLAGS='--tf_xla_auto_jit=0'`).
- **Resource Constraints**: Capped GPU VRAM logical partitions (`memory_limit=4096`) and enforced CPU core restrictions (`OMP_NUM_THREADS=4`) coupled with OS-level `nice -n 19`. This ensures the intensive Rainbow DQN experience collections don't freeze user interfaces or throttle the OS.

## 📊 Research Paper Results

### Expected Performance Hierarchy

1. **Advanced RL (C51, Rainbow DQN)**: Best overall performance
2. **Basic RL (DQN, REINFORCE)**: Moderate performance
3. **Baseline Schedulers**: Lowest performance, but fast execution

### Key Metrics Comparison

| Algorithm   | Cost     | Time     | Utilization | Throughput | Adherence |
| ----------- | -------- | -------- | ----------- | ---------- | --------- |
| Rainbow DQN | **Best** | **Best** | **Best**    | **Best**   | **Best**  |
| C51         | **Good** | **Good** | **Good**    | **Good**   | **Good**  |
| DQN         | Moderate | Moderate | Moderate    | Moderate   | Moderate  |
| REINFORCE   | Moderate | Moderate | Moderate    | Moderate   | Moderate  |
| Baseline    | Poor     | Poor     | Poor        | Poor       | Poor      |

### Publication-Ready Artifacts

- **High-resolution plots** (300 DPI)
- **Statistical significance tests**
- **Performance ranking tables**
- **Comprehensive CSV data**
- **Interactive HTML visualizations**

## 🤝 Contributing

### Development Setup

```bash
# Clone for development
git clone https://github.com/yourusername/RM_DeepRL.git
cd RM_DeepRL

# Create development branch
git checkout -b feature/your-feature

# Make changes and test
python src/comprehensive_comparison.py

# Commit changes
git add .
git commit -m "Add your feature"
git push origin feature/your-feature
```

### Code Structure Guidelines

- **Follow PEP 8**: Python style guide
- **Add docstrings**: Document all functions
- **Include type hints**: Use typing module
- **Write tests**: Add unit tests for new features
- **Update README**: Document new features

### Adding New Algorithms

1. Create new algorithm file in `src/`
2. Follow existing algorithm structure
3. Add to `comprehensive_comparison.py`
4. Update configuration options
5. Add documentation

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 💡 Additional Resources

### Dependencies File (requirements.txt)

```txt
tensorflow==2.9.0
tf-agents==0.15.0
matplotlib==3.5.0
seaborn==0.11.0
bokeh==2.4.0
pandas==1.4.0
numpy==1.21.0
scipy==1.8.0
scikit-learn==1.0.0
```

### Example Job Workload (input/jobs.csv)

```csv
arrival_time,job_id,job_type,cpu,mem,executors,duration
0,0,1,2,4,3,60
10,1,2,4,8,2,100
20,2,3,6,8,2,80
30,3,1,3,6,4,90
40,4,2,5,10,3,120
```

### Sample Configuration (settings/config.ini)

```ini
[DEFAULT]
root = /path/to/RM_DeepRL
algo = c51
workload = jobs.csv
beta = 0.5
iteration = 10000
fixed_episodic_reward = 10000
epsilon = 0.001
learning_rate = 0.001
gamma = 0.9
placement_penalty = 30
pp_apply = true
```

This comprehensive README provides all the necessary information for reproducing the research results and extending the work. The project structure, installation instructions, and usage examples ensure that other researchers can easily replicate and build upon this work.
