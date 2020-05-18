from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='swish_cpp',
      ext_modules=[cpp_extension.CppExtension('swish_cpp', ['swish.cc'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
