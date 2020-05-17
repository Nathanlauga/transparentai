from setuptools import setup, find_packages
import os

path = os.path.dirname(os.path.abspath(__file__))
with open(path+"/README.md", "r") as fh:
    long_description = fh.read()
fh.close()

setup(
    name='transparentai',
    version='0.2.1',
    description="Python tool to create or inspect a transparent and ethical AI.",
    license='MIT',
    author='Nathan LAUGA',
    author_email='nathan.lauga@protonmail.com',
    url='https://github.com/Nathanlauga/transparentai',
    packages=[pkg for pkg in find_packages(
    ) if pkg.startswith('transparentai')],
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'shap',
        'energyusage',
        'safety',
        'seaborn',
        'ipywidgets'
    ],
    long_description_content_type="text/markdown",
    long_description=long_description,
    python_requires='>=3.6',
    classifiers=[
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    entry_points={

    },
)
