from setuptools import setup, find_packages

setup(
    name="auglift",
    version="0.1.0",
    description="AugLift: Depth-Augmented 3D Human Pose Lifting",
    author="AugLift Team",
    python_requires=">=3.10",
    packages=find_packages(include=["mmpose*", "coarse_depth_experiments*", "auglift*", "configs*"]),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "torchaudio",
        "openmim",
        "mmengine",
        "mmcv>=2.0.0",
        "mmpose>=1.2.0",
        "numpy",
        "opencv-python",
        "scipy",
        "matplotlib",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
        ],
    },
)
