# MPTS-Sinusoid

This repository implements the MPTS and baselines proposed in the paper **"Beyond Any-Shot Adaptation: Predicting Optimization Outcome for Robustness Gains without Extra Pay"**, focusing on sinusoid regression scenario.

## 🚀 Quick Start

### Installation
```bash
conda create -n mpts_sinu python=3.7 -y
conda activate mpts_sinu

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Benchmarking All Methods
### Command Line Examples
| Method | Command |
|--------|---------|
| **MPTS** (Ours)   | `python main.py --gpu_id 0 --log_name logs/mpts --sampling_strategy mpts --num_candidates 32 --global_seed 1` |
| ERM    | `python main.py --gpu_id 0 --log_name logs/erm --sampling_strategy erm --num_candidates 16 --global_seed 1` |
| DRM    | `python main.py --gpu_id 0 --log_name logs/drm --sampling_strategy drm --num_candidates 32 --global_seed 1` |
| GDRM   | `python main.py --gpu_id 0 --log_name logs/gdrm --sampling_strategy gdrm --num_candidates 16 --global_seed 1` |

