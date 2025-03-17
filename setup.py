from setuptools import setup, find_packages

setup(
    name="uls_sam_mlx",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mlx",
        "numpy",
        "opencv-python",
        "tqdm",
        "matplotlib",
        "requests"
    ],
    author="Klaudiusz",
    author_email="example@example.com",
    description="High-performance SAM2.1 implementation for Apple Silicon using MLX framework",
    keywords="segmentation, medical-imaging, ultrasonography, apple-silicon, mlx",
    url="https://github.com/Szowesgad/ULS-Sam-MLX",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)