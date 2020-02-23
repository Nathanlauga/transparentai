from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
fh.close()

setup(
    name='transparentai',
    version='0.0.2',
    description="Python tool to create an ethic AI.",
    license='MIT',
    author='Nathan LAUGA',
    author_email='nathan.lauga@protonmail.com',
    url='https://github.com/Nathanlauga/transparentai',
    packages=[pkg for pkg in find_packages() if pkg.startswith('transparentai')],
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn',
        'ipython'
    ],
    long_description=long_description,
    python_requires='>=3.5',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    entry_points = {

    },
)
