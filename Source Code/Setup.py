"""
Setup.py
Alexander Marsh
Last Edit 10 September 2025

GNU Affero General Public License

Python file that builds the c++ LibTorch module into a .so PyTorch module.
"""

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='LooseTopologicalNode',
    ext_modules=[
        CppExtension(
            name='LooseTopologicalNode',
            sources=[
                'Bindings.cpp',             # Pybind11 bindings
                'TopologicalNodes.cpp'      # C++ implementation
            ],
            include_dirs=['.'],             # Header files are in the current directory
            extra_compile_args=['-O3'],     # Optimization flag
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
