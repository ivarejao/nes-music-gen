This files is for helping running on Infra of the lab

```
apptainer pull nesmdb.sif docker://ivarejao/nesmdb_api:v1.0
```

Kernels are installed on `/local/$USER/.local/share/kernels/nesenv`

```
# uv environment on /tmp
UV_PROJECT_ENVIRONMENT="/tmp/$USER/nesenv"
UV_CACHE_DIR="/tmp/$USER/uv-cache"
UV_PYTHON_INSTALL_DIR="/tmp/$USER/uv-python"
# ffmpeg
PATH="/tmp/$USER/ffmpeg-runtime/bin:$PATH"
LD_LIBRARY_PATH="/tmp/$USER/ffmpeg-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

In order to install ffmpeg
```
module avail conda # check if anaconda3 is available at the server
module load anaconda3

mkdir -p "/tmp/$USER/conda-pkgs"

CONDA_PKGS_DIRS="/tmp/$USER/conda-pkgs" \
conda create \
  --prefix "/tmp/$USER/ffmpeg-runtime" \
  --channel conda-forge \
  --override-channels \
  "ffmpeg=7" \
  --yes
```

Make ffmpeg available at your environment
```
export PATH="/tmp/$USER/ffmpeg-runtime/bin:$PATH"
export LD_LIBRARY_PATH="/tmp/$USER/ffmpeg-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ffmpeg -version
```