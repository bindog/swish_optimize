from setuptools import setup, Extension
from torch.utils import cpp_extension
import sysconfig
# please check this path and make sure it exists
# if not exists, please install python3.x-dev or other related libs
# print(sysconfig.get_paths()['include'])

EXT_SRCS = ['csrc/swish.cc', 'csrc/swish_cuda_kernel.cu']

setup(
    name='swish_cpp',
    version='0.9.9',
    install_requires=['torch>=1.2'],
    ext_modules=[cpp_extension.CUDAExtension(
        'swish_cpp',
        EXT_SRCS,
        extra_compile_args={
            'cxx': [],
            'nvcc': ['--expt-extended-lambda']
        },
        include_dirs=[]
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
