from setuptools import setup, Extension
from torch.utils import cpp_extension
import sysconfig
# please check this path and make sure it exists
# if not exists, please install python3.x-dev or other related libs
# print(sysconfig.get_paths()['include'])

setup(name='swish_cpp',
      ext_modules=[cpp_extension.CppExtension('swish_cpp', ['swish.cc'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
