# Diffraction-Dragrace
A repo for speed comparisons of elementary operations used in optical simulation

# Installation
You can set up the environment by running the following commands:

```bash
cd Diffraction-Dragrace
conda activate base
conda create -n dragrace "python=3.11" --file requirements.txt --channel conda-forge --override-channels
conda activate dragrace
pip install -e .
```

Installing three machine learning libraries takes a few minutes:
- PyTorch
- TensorFlow
- Jax
