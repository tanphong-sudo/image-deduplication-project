from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import platform

def get_pybind_include():
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        return None


# Định nghĩa C++ extension module
ext_modules = [
    Extension(
        name='lsh_cpp_module',
        sources=['lsh_cpp/bindings.cpp'],
        include_dirs=[
            # Thư mục chứa pybind11 headers
            get_pybind_include() or '',
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
            '-ffast-math',     # Fast math operations
            '-Wall',           # Enable warnings
            '-Wextra',         # Extra warnings
        ]
        
        if platform.system() == 'Darwin':  # macOS
            machine = platform.machine()
            if machine == 'arm64':
                # Apple Silicon - use generic optimization
                args.append('-mcpu=apple-m1')  # works for M1/M2/M3
            else:
                # Intel Mac
                args.append('-march=native')
            # Skip OpenMP on macOS (requires libomp: brew install libomp)
            # Uncomment below if you have libomp installed:
            # args.append('-Xpreprocessor')
            # args.append('-fopenmp')
        else:  # Linux
            args.append('-march=native')
            args.append('-fopenmp')
    
    return args


def get_link_args():
    """Lấy linker arguments."""
    args = []
    
    if platform.system() == 'Darwin':  # macOS
        # Skip OpenMP linking on macOS (requires: brew install libomp)
        # Uncomment if you have libomp installed:
        # args = ['-lomp']
        pass
    elif platform.system() == 'Linux':
        args = ['-fopenmp']
    
    return args


# Apply compile and link args
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
