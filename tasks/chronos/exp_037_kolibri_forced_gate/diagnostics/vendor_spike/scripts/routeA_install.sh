#!/bin/bash
# Route A install attempt: vLLM 0.29.0 from source (macOS CPU) + vendor plugin, in a dedicated venv.
# Deadline: 30 min from spike start (04:07:20Z) -> 04:37:20Z. Kills itself at the deadline.
set -x
export PYTHONDONTWRITEBYTECODE=1
V="${SPIKE_DIR:?set SPIKE_DIR to the scratch folder of the spike}"  # exp_037 copy: was the scratch path on the mini
VL=$V/downloads/src/vllm-0.29.0
VA=$V/downloads/src/aleph-alpha-inference-049a6a7bd2405b27d6d280d256bd3d585191c7ae
export UV_CACHE_DIR=$V/uv-cache
cd $V/routeA
date -u
uv venv --python 3.12 $V/venvA || exit 2
PY=$V/venvA/bin/python
uv pip install --python $PY "torch==2.13.0" "cmake>=3.26.1" ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" wheel jinja2 || exit 3
date -u
uv pip install --python $PY -r $VL/requirements/cpu.txt || exit 4
date -u
export SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0
export VLLM_TARGET_DEVICE=cpu
export PATH=$V/venvA/bin:$PATH
export MAX_JOBS=12
uv pip install --python $PY --no-build-isolation --no-deps $VL || exit 5
date -u
uv pip install --python $PY --no-build-isolation --no-deps $VA || exit 6
date -u
cd $V/routeA && $PY -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__); import aleph_alpha_inference.kolibri1 as k; print('vendor import OK', k.__file__)" || exit 7
date -u
echo ROUTE_A_IMPORT_OK
