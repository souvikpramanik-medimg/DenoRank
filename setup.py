from setuptools import setup, find_packages

setup(
    name="denorank",
    version="1.0.0",
    author="Souvik Pramanik",
    author_email="spramanik.diat23@gmail.com",
    description="Gradient-free synthetic sample selection for low-label nuclei segmentation "
                "via diffusion denoising residuals",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/souvikpramanik-medimg/DenoRank",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0", "torchvision>=0.15.0", "numpy>=1.23.0", "scipy>=1.9.0",
        "pandas>=1.5.0", "Pillow>=9.0.0", "opencv-python-headless>=4.6.0",
        "matplotlib>=3.5.0", "tqdm>=4.64.0", "diffusers>=0.27.0",
        "transformers>=4.38.0", "accelerate>=0.27.0", "peft>=0.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords=["nuclei segmentation", "synthetic data", "data selection", "stable diffusion",
              "LoRA", "TracIn", "histopathology", "low-label learning"],
)
