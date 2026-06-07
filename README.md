# pascal-pkgs-ci

The main repository for building Pascal-compatible versions of ML applications and libraries.

1. vLLM `0.5.5`, `0.6.0`, `0.6.1`, `0.6.1.post1`, `0.6.1.post`, `0.6.2`, `0.6.3`, `0.6.3.post1`, `0.6.4`, `0.6.4.post1`, `0.6.5`, `0.6.6`, `0.6.6.post1`, `0.7.0`, `0.7.1`, `0.7.2`, `0.7.3`, `0.8.0`, `0.8.1`, `0.8.2`, `0.8.3`, `0.8.4`, `0.8.5`, `0.9.0`, `0.9.1`, and `main` (nightly, updates daily) are available in this repository.
2. Triton `2.2.0`, `2.3.0`, `2.3.1`, `3.0.0`, `3.1.0`, `3.2.0`, `3.3.0`, `3.3.1`, `3.4.0` are available in this repository.

> [!IMPORTANT]
> **WARNING:** Support for new GPUs has been disabled (`v0.7.0`+/`main`)
> 
> Due to the increase in vLLM code amount, binary size, and build speed, it is now impractical to build vLLM for all GPU architectures.  
> To use vLLM on a heterogeneous machine or cluster, use the official version of vLLM for non-Pascal GPUs and this version for Pascal GPUs and use tensor or pipeline parallelism to connect instances.
> 
> Note that this change only affects versions above `v0.7.0` (including `main`).  

## Installation (docker)

### [vllm](https://github.com/vllm-project/vllm)

```sh
# Pull the vLLM image
docker pull ghcr.io/sasha0552/vllm:v0.10.0  # you can omit the version specifier
                                            # to install nightly version

# You can now follow the official vLLM documentation.
# Replace the official image with this one.
```

## Installation (manual)

### Pull vLLM source

```bash
git clone https://github.com/vllm-project/vllm
cd vllm
#checkout the desired branch/tag 
git checkout releases/v0.9.1
# install dependencies
uv sync
# Transformers need to be installed separately with the version `transformers==4.51.3`
pip install "transformers==4.51.3" "huggingface-hub==0.36.2" "okenizers==0.21.4"

```

> [!WARNING]
> Wheels, as of v0.6.5, is currently in a soft-broken state due to PyTorch.
> To use them, you need to manually patch PyTorch after installation of vLLM.
>
> <details>
> <summary>Patching PyTorch</summary>
>
> Example command assuming you are using a virtual environment located in the current directory
>
> ```sh
> sed -e "s/.major < 7/.major < 6/g"                                 \
>     -e "s/.major >= 7/.major >= 6/g"                               \
>     -i                                                             \
>     venv/lib/python3.12/site-packages/torch/_inductor/scheduler.py \
>     venv/lib/python3.12/site-packages/torch/utils/_triton.py
> ```
> </details>

I recommend installing [transient-package](https://pypi.org/project/transient-package) before proceeding. It simplifies the installation of `triton`.

You can install it globally with `pipx`:

```sh
pipx install transient-package
```

> [!IMPORTANT]
> <details>
> <summary>If you don't want to install transient-package</summary>
>
> If you don't want to install `transient-package`, you'll need to replace
>
> ```sh
> transient-package install       \
>   --interpreter venv/bin/python \
>   --source triton               \
>   --target triton-pascal
> ```
>
> with
>
> ```sh
> # Remove triton
> pip uninstall triton
>
> # Install patched triton
> pip install triton-pascal
> ```
>
> Note that `transient-package` does more than just `pip uninstall triton` and `pip install triton-pascal`.
> In particular, it tries to install the correct version of `triton`, and creates a bogus `triton` package in case the application checks for the presence of `triton`.
> </details>

### [vllm](https://github.com/vllm-project/vllm)

```sh
# Use this repository
export PIP_EXTRA_INDEX_URL="https://sasha0552.github.io/pascal-pkgs-ci/"

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install vLLM
pip3 install vllm-pascal==0.10.0  # you can omit the version specifier
                                  # to install nightly version

# Install patched triton
transient-package install       \
  --interpreter venv/bin/python \
  --source triton               \
  --target triton-pascal


# Launch vLLM
vllm serve --help
```

If using `conda`

```sh
# Use this repository
export PIP_EXTRA_INDEX_URL="https://sasha0552.github.io/pascal-pkgs-ci/"

# Create conda virtual environment
conda create -n vllm-env python

# Activate virtual environment
conda activate vllm-env
conda install uv pipx --channel conda-forge -y

# Install vLLM
pip install vllm-pascal==0.10.0  # you can omit the version specifier
                                  # to install nightly version

# Install patched triton
transient-package install       \
  --interpreter $CONDA_PREFIX/bin/python \
  --source triton               \
  --target triton-pascal


# Launch vLLM
vllm serve --help
```

Notes: Fix the ***"vLLM package not found"*** warning permanently

This is what breaks platform detection. Create a fake dist-info so importlib.metadata finds vllm:

```bash
# Find where vllm-pascal installed its files
VLLM_DIR=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Get the actual version
VLLM_VERSION=$(python3 -c "from importlib.metadata import version; print(version('vllm-pascal'))" 2>/dev/null || echo "0.9.1")

# Create a minimal dist-info directory that claims to be 'vllm'
mkdir -p "$SITE_PACKAGES/vllm-${VLLM_VERSION}.dist-info"

cat > "$SITE_PACKAGES/vllm-${VLLM_VERSION}.dist-info/METADATA" << EOF
Metadata-Version: 2.1
Name: vllm
Version: ${VLLM_VERSION}
EOF

cat > "$SITE_PACKAGES/vllm-${VLLM_VERSION}.dist-info/RECORD" << EOF
EOF

echo "Created vllm dist-info for version $VLLM_VERSION"
```

### [aphrodite-engine](https://github.com/PygmalionAI/aphrodite-engine)

```sh
# Use this repository
export PIP_EXTRA_INDEX_URL="https://sasha0552.github.io/pascal-pkgs-ci/"

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install aphrodite-engine
pip3 install --extra-index-url https://downloads.pygmalion.chat/whl aphrodite-engine

# Install patched triton
transient-package install       \
  --interpreter venv/bin/python \
  --source triton               \
  --target triton-pascal

# Launch aphrodite-engine
aphrodite --help
```

### [triton](https://github.com/triton-lang/triton) (for other applications)

```sh
# Use this repository
export PIP_EXTRA_INDEX_URL="https://sasha0552.github.io/pascal-pkgs-ci/"

# Install patched triton
transient-package install       \
  --interpreter venv/bin/python \
  --source triton               \
  --target triton-pascal
```

If using `conda`

```sh
transient-package install       \
  --interpreter $CONDA_PREFIX/bin/python \
  --source triton               \
  --target triton-pascal
```

---

<details>
<summary>Instructions for uploading to PyPI</summary>

```sh
# Download artifacts
gh run download <run id>

# Install twine
pip3 install twine

# Upload wheels
TWINE_PASSWORD=<pypi token> twine upload */*.whl
```
</details>
