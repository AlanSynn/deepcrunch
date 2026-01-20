import glob
import os
import os.path as osp
import platform
import sys
from itertools import product

import torch
from setuptools import find_packages, setup
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

__version__ = "0.1.0"
URL = "https://github.com/AlanSynn/deepcrunch"

# Check if CUDA is available
WITH_CUDA = False
if torch.cuda.is_available():
    WITH_CUDA = CUDA_HOME is not None or torch.version.hip

# Determine suffices based on CUDA availability
suffices = ["cpu", "cuda"] if WITH_CUDA else ["cpu"]

# Check if FORCE_CUDA environment variable is set to "1"
if os.getenv("FORCE_CUDA", "0") == "1":
    suffices = ["cuda", "cpu"]

# Check if FORCE_ONLY_CUDA environment variable is set to "1"
if os.getenv("FORCE_ONLY_CUDA", "0") == "1":
    suffices = ["cuda"]

# Check if FORCE_ONLY_CPU environment variable is set to "1"
if os.getenv("FORCE_ONLY_CPU", "0") == "1":
    suffices = ["cpu"]

BUILD_DOCS = os.getenv("BUILD_DOCS", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"


def get_extensions():
    extensions = []

    extensions_dir = osp.join("csrc")
    main_files = glob.glob(osp.join(extensions_dir, "*.cc"))
    # Remove generated 'hip' files, in case of rebuilds
    main_files = [path for path in main_files if "hip" not in path]

    for main, suffix in product(main_files, suffices):
        define_macros = [("WITH_PYTHON", None)]
        undef_macros = []

        if sys.platform == "win32":
            define_macros += [("deepcrunch_EXPORTS", None)]

        extra_compile_args = {"cxx": ["-O3"]}
        if not os.name == "nt":  # Not on Windows:
            extra_compile_args["cxx"] += ["-Wno-sign-compare"]
        extra_link_args = [] if WITH_SYMBOLS else ["-s"]

        info = parallel_info()
        if (
            "backend: OpenMP" in info
            and "OpenMP not found" not in info
            and sys.platform != "darwin"
        ):
            extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
            if sys.platform == "win32":
                extra_compile_args["cxx"] += ["/openmp"]
            else:
                extra_compile_args["cxx"] += ["-fopenmp"]
        else:
            print("Compiling without OpenMP...")

        # As of PyTorch 2.0.1, quantization is not supported on macOS arm64.
        # See https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/quantized/cpu/qnnpack
        if sys.platform == "darwin" and platform.machine() == "arm64":
            print("PyTorch does not support quantization on macOS arm64.")

        if suffix == "cuda":
            define_macros += [("WITH_CUDA", None)]
            nvcc_flags = os.getenv("NVCC_FLAGS", "")
            nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
            nvcc_flags += ["-O3"]
            if torch.version.hip:
                # USE_ROCM was added to later versions of PyTorch.
                # Define here to support older PyTorch versions as well:
                define_macros += [("USE_ROCM", None)]
                undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
            else:
                nvcc_flags += ["--expt-relaxed-constexpr"]
            extra_compile_args["nvcc"] = nvcc_flags

        name = main.split(os.sep)[-1][:-4]
        sources = [main]

        path = osp.join(extensions_dir, "cpu", f"{name}_cpu.cpp")
        if osp.exists(path):
            sources += [path]

        path = osp.join(extensions_dir, "cuda", f"{name}_cuda.cu")
        if suffix == "cuda" and osp.exists(path):
            sources += [path]

        Extension = CppExtension if suffix == "cpu" else CUDAExtension
        extension = Extension(
            f"deepcrunch._{name}_{suffix}",
            sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
        extensions += [extension]

    return extensions


install_requires = [
    "neural-compressor",
]

test_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-timeout>=2.1.0",
    "pytest-xdist>=3.0.0",  # For parallel test execution
    "numpy>=1.21.0",
    "onnx>=1.12.0",
    "onnxruntime>=1.12.0",
    "torch>=1.12.0",
    "torchvision>=0.13.0",
]

# Workaround for hipify abs paths
include_package_data = True
if torch.cuda.is_available() and torch.version.hip:
    include_package_data = False

setup(
    name="deepcrunch",
    version=__version__,
    description="PyTorch Extension Library of Model Compression for General Usage",
    author="Alan Synn",
    author_email="alan@alansynn.com",
    url=URL,
    download_url=f"{URL}/archive/{__version__}.tar.gz",
    keywords=[
        "pytorch",
        "model compression",
        "quantization",
        "pruning",
        "distillation",
        "quantization aware training",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
    },
    ext_modules=get_extensions() if not BUILD_DOCS else [],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            no_python_abi_suffix=True, use_ninja=False
        )
    },
    packages=find_packages(),
    include_package_data=include_package_data,
)
