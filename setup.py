#!/usr/bin/env python3
"""
Setup script for the Physiological Anomaly Detection package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
README_PATH = Path(__file__).parent / "README.md"
with open(README_PATH, "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
    requirements = [
        line.strip() 
        for line in f.readlines() 
        if line.strip() and not line.startswith("#")
    ]

# Package metadata
PACKAGE_NAME = "physio_anomaly_detector"
VERSION = "1.0.0"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your.email@example.com"
DESCRIPTION = "Advanced physiological anomaly detection system"
URL = "https://github.com/yourusername/anomaly_detector_physio"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.0",
        ],
        "monitoring": [
            "mlflow>=2.5.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "database": [
            "sqlalchemy>=2.0.0",
            "psycopg2-binary>=2.9.0",
            "pymysql>=1.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "physio-train=train_eval:main",
            "physio-advanced-train=advanced_train_eval:main",
            "physio-api=api.main:main",
            "physio-download-wesad=data.get_wesad:main",
            "physio-prepare-wesad=data.prepare_wesad_csv:main",
        ],
    },
    package_data={
        "": ["config/*.yaml", "config/*.yml"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "anomaly detection",
        "physiological signals",
        "machine learning",
        "deep learning",
        "time series",
        "healthcare",
        "wearables",
        "EDA",
        "heart rate",
        "stress detection",
    ],
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
        "Documentation": f"{URL}/docs",
    },
)
