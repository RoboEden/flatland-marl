import sys
from glob import glob

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit

ext_modules = [
    Pybind11Extension(
        "flatland_cutils",
        # ["src/main.cpp"],
        sorted(glob("src/*.cpp")),
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
        # extra_compile_args = ["-O0", "-g", "-ggdb3"],
    ),
]

setup(
    name="flatland_cutils",
    version=__version__,
    author="Yuhao",
    author_email="yuhaojiang@chaocanshu.ai",
    url="https://gitlab.parametrix.cn/parametrix/challenge/flatland-ccs/tree/yuhao_v3",
    description="C",
    long_description="",
    ext_modules=ext_modules,
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
