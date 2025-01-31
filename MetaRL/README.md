# MPTS-MetaRL

This repository implements the MPTS and baselines proposed in the paper **"Beyond Any-Shot Adaptation: Predicting Optimization Outcome for Robustness Gains without Extra Pay"**, focusing on Meta-RL scenarios. The code is adapted from https://github.com/lmzintgraf/cavia/tree/master/rl .

## 🚀 Quick Start

### Installation
```bash
conda create -n mpts_metarl python=3.7 -y
conda create -n mpts_metarl

# Install dependencies (mujoco required)
pip install -r requirements.txt
```

## 🔧 Benchmarking All Methods
### Command Line Examples
| Method | Command |
|--------|---------|
| **MPTS**(Ours)   | `python main.py --env-name={env_name} --game_framework=2 --seed=10 --sampling_strategy=mp --add_random --sample_ratio=1.5` |
| ERM    | `python main.py --env-name={env_name} --game_framework=2 --seed=10 --sampling_strategy=random` |
| DRM    | `python main.py --env-name={env_name} --game_framework=1 --seed=10 --sampling_strategy=random` |
| GDRM   | `python main.py --env-name={env_name} --game_framework=3 --seed=10 --sampling_strategy=random` |

**Supported Environments:**
- HalfCheetahMassVel-v1
- HalfCheetahVel-v0
- Walker2dMassVel-v1
- Walker2dVel-v1
- ReacherPos-v0

More scenarios can be easily included in `envs/`.