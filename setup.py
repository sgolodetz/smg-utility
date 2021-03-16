from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="smg-utility",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Various helper classes and functions",
    long_description="",  #long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-utility",
    packages=find_packages(include=["smg.utility", "smg.utility.*"]),
    include_package_data=True,
    install_requires=[
        "future-annotations",
        "numba",
        "numpy",
        "opencv-contrib-python==3.4.2.16",
        "pytypes @ git+https://github.com/sgolodetz/pytypes.git"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
