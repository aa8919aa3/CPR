"""
Setup script for CPR - Current-Phase Relation
"""
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="cpr-josephson-analysis",
    version="1.0.0",
    description="High-performance Josephson junction current-phase relation analysis suite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aa8919aa3/CPR",
    author="CPR Team",
    author_email="support@cpr.dev",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="josephson junction, superconductivity, physics, data analysis, numba, parallel processing",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "astropy>=5.0.0",
        "numba>=0.56.0",
        "psutil>=5.8.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "pre-commit>=2.15.0",
        ],
        "performance": [
            "fireducks-pandas",
        ],
    },
    entry_points={
        "console_scripts": [
            "cpr-process=cpr.main_processor:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/aa8919aa3/CPR/issues",
        "Source": "https://github.com/aa8919aa3/CPR/",
        "Documentation": "https://github.com/aa8919aa3/CPR/wiki",
    },
)
