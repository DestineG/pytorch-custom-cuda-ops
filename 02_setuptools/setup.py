from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="online_softmax_speedup", # 安装后的包名
    ext_modules=[
        CUDAExtension(
            name="online_softmax_cuda", # 编译生成的 .so 模块名
            sources=[
                "csrc/binding.cpp",
                "csrc/softmax_kernel.cu",
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension # 使用 PyTorch 提供的后端
    }
)