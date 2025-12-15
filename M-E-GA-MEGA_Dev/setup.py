from setuptools import setup, find_packages

setup(
    name="M-E-GA2",
    version="2.0.0-b6",

    author="Matt Andrews",
    author_email="Matthew.Andrews2024@gmail.com",
    description="A Genetic Algorithm framework with Mutable Encoding.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ML-flash/M-E-GA",
    packages=find_packages(where="src"),  # <== Finds packages inside "src/"
    package_dir={"": "src"},  # <== Tells setuptools to use "src/" as the base directory
    include_package_data=True,
    install_requires=[
        "xxhash",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
