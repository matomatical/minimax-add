minimax-add
-----------

JAX reproduction of **Adversarial Environment Design via Regret-Guided
Diffusion Models** (Chung et al., 2024), built on top of
[minimax](https://github.com/facebookresearch/minimax) (Jiang et al., 2023).

This is a fork of minimax that adds:

- Modern JAX compatibility (JAX >= 0.6, distrax replacing tfp)

- The ADD algorithm (`minimax.add`) including DDPM/DDIM diffusion, an
  environment critic, and regret-guided level sampling for the maze domain

The original minimax code is available under Apache 2.0 license.

## Installation

Requires Python >= 3.12.

```bash
git clone <this-repo>
cd minimax-add
pip install -e .
```

For TPU support:
```bash
pip install jax[tpu]
```


## Usage

ADD has three training stages:

```bash
# Stage 1: Pretrain diffusion model on random maze levels
python -m minimax.add.train_diffusion

# Stage 2: RL with diffusion-sampled levels (no guidance)
python -m minimax.add.train_rl --diffusion_ckpt checkpoints/diffusion/final.pkl

# Stage 3: Full ADD with regret-guided diffusion
python -m minimax.add.train --diffusion_ckpt checkpoints/diffusion/final.pkl
```

All training scripts accept `--help` for full argument documentation.


## Package structure

```
src/minimax/
  add/                  ADD implementation
    runner.py           ADDRunner (extends DRRunner)
    unet.py             Denoising UNet for DDPM
    diffusion.py        DDPM forward process, DDIM sampling
    theta.py            Level encoding/decoding (maze <-> image)
    guidance.py         Regret-guided DDIM sampling
    critic.py           Distributional environment critic
    train.py            Stage 3 training loop
    train_diffusion.py  Stage 1 training loop
    train_rl.py         Stage 2 training loop
  runners/              Minimax UED runners (DR, PLR, PAIRED, ...)
  envs/                 JAX environments (maze, ...)
  models/               RL model architectures
  agents/               PPO agent
```


## References

- Chung et al., "Adversarial Environment Design via Regret-Guided Diffusion
  Models," NeurIPS 2024
  [[arXiv](https://arxiv.org/abs/2410.19715)]
  [[repo (PyTorch)](https://github.com/rllab-snu/ADD)]

- Jiang et al., "minimax: Efficient Baselines for Autocurricula in JAX," 2023
  [[arXiv](https://arxiv.org/abs/2311.12716)]
  [[repo](https://github.com/facebookresearch/minimax)]
