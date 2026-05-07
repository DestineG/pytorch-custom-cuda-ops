#!/bin/bash
set -e  # 出错立即停止

mkdir -p build && cd build

TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

cmake -DCMAKE_PREFIX_PATH="$TORCH_PATH" ..

make -j$(nproc)

cp *.so ..

echo "---------------------------------------"
echo "Build Successful: online_softmax_cuda.so is ready!"