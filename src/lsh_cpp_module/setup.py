from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import platform

class get_pybind_include(object):
    """Helper class để lấy pybind11 include directory."""
    
    def __str__(self):
        import pybind11
        return pybind11.get_include()


# Định nghĩa C++ extension module
ext_modules = [
    Extension(
        name='lsh_cpp_module',
        sources=['lsh_cpp/bindings.cpp'],
        include_dirs=[
            # Thư mục chứa pybind11 headers
            get_pybind_include(),
            # Thư mục hiện tại (để include simhash.cpp)
            'lsh_cpp',
        ],
        language='c++',
        extra_compile_args=[],
        extra_link_args=[],
    ),
]


def get_compile_args():
    """Lấy compiler arguments phù hợp với hệ điều hành."""
    args = []
    
    if platform.system() == 'Windows':
        # MSVC compiler flags
        args = ['/std:c++14', '/O2']
    else:
        # GCC/Clang compiler flags
        args = [
            '-std=c++14',      # C++14 standard
            '-O3',             # Optimization level 3 (maximum speed)
            '-march=native',   # Optimize cho CPU hiện tại
            '-ffast-math',     # Fast math operations
            '-Wall',           # Enable warnings
            '-Wextra',         # Extra warnings
        ]
        
        # Thêm OpenMP nếu có (parallel processing)
        if platform.system() == 'Darwin':  # macOS
            # macOS 
            args.append('-Xpreprocessor')
            args.append('-fopenmp')
        else:  # Linux
            args.append('-fopenmp')
    
    return args


def get_link_args():
    """Lấy linker arguments."""
    args = []
    
    if platform.system() == 'Darwin':  # macOS
        # Link với OpenMP library nếu có
        args = ['-lomp']
    elif platform.system() == 'Linux':
        args = ['-fopenmp']
    
    return args


# Apply compile và link args
ext_modules[0].extra_compile_args = get_compile_args()
ext_modules[0].extra_link_args = get_link_args()


class BuildExt(build_ext):
    """Custom build extension để handle C++14/17 compilation."""
    
    def build_extensions(self):
        # Compiler-specific options
        ct = self.compiler.compiler_type
        opts = []
        
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            
        for ext in self.extensions:
            ext.extra_compile_args = opts + ext.extra_compile_args
            
        build_ext.build_extensions(self)


# Setup configuration
setup(
    name='lsh_cpp_module',
    version='1.0.0',
    author='',
    author_email='',
    description='High-performance LSH (SimHash) implementation in C++',
    long_description='''
    A high-performance Locality-Sensitive Hashing (LSH) implementation using SimHash algorithm.
    Built with C++ for speed and exposed to Python via Pybind11.
    
    Features:
    - Fast SimHash computation for text
    - Random projection based LSH for high-dimensional vectors
    - Efficient nearest neighbor search
    - Multi-table hashing for better recall
    - Batch operations support
    ''',
    ext_modules=ext_modules,
    install_requires=[
        'pybind11>=2.6.0',
        'numpy>=1.19.0',
    ],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.6',
)
