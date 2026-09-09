# Local setup

The tested machine uses Python 3.12 and a GTX 1050 with 3 GiB of memory (sm_61).
The renderer is 2DGS's `diff-surfel-rasterization`; the attempted gsplat builds did not
support this card. No custom CUDA kernels are part of this project.

## 1. Create the environment

Run these commands from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install the tested PyTorch pair

```sh
python -m pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

The local build used CUDA 12.6 nvcc and driver 565.57 with these cu124 wheels.
Keep torch and torchvision paired when reproducing this environment.

## 3. Build the renderer

Place `hbb1/2d-gaussian-splatting` under `baselines/2d-gaussian-splatting/`, checked
out at `335ad612f2e783a4e57b9cbc4d1e167bd599fc98`, including its submodules.

The local GCC 13 build needed two missing includes: `<cstdint>` in the rasterizer's
`cuda_rasterizer/rasterizer_impl.h`, and `<cfloat>` in simple-knn's `simple_knn.cu`.

```sh
TORCH_CUDA_ARCH_LIST="6.1" MAX_JOBS=4 python -m pip install --no-build-isolation \
    baselines/2d-gaussian-splatting/submodules/diff-surfel-rasterization \
    baselines/2d-gaussian-splatting/submodules/simple-knn
```

The architecture flag is for the GTX 1050. A different GPU needs its own target;
this setup has only been checked on the local machine.

## 4. Install the package and check it

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python -m pytest -q
python scripts/env_check.py
```

The tests run on CPU. The environment check also exercises CUDA and the renderer.
Datasets and checkpoints are separate local files; see the [README](../README.md#running-it)
for the expected paths and training commands.
