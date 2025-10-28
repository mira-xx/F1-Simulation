from setuptools import setup, find_packages

setup(
    name="strategy-sim",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "fastf1",
        "numpy",
        "pandas",
        "statsmodels",
    ],
    python_requires=">=3.8",
)