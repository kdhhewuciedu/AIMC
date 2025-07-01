export CXX=/usr/bin/g++
export CC=/usr/bin/gcc
export MKLROOT=$CONDA_PREFIX
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
# export CUDA_VERSION=11.3
export CUDA_VERSION=11.8
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
export CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}
export CUDA_LIB_PATH=${CUDA_HOME}/lib64
export CUDA_INCLUDE_DIRS=${CUDA_HOME}/include
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# export PYTHONPATH=../../../aihwkit/src/:$PYTHONPATH
export PYTHONPATH=../aihwkit/src/:$PYTHONPATH