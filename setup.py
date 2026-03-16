from setuptools import setup, find_packages

setup(
    name="distributed-hcvgloc",
    version="0.1.0",
    description="Distributed Training Study for Hierarchical Cross-View Geo-Localization",
    author="Ganesh (gs37r)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "timm>=0.9.12",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
    ],
)
