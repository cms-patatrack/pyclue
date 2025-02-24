
import sys
from pathlib import Path
from setuptools import setup
import subprocess

__version__ = "2.6.3"

this_directory = Path(__file__).parent
long_description = (this_directory/'README.md').read_text()

cmake_command = ['cmake', '-B', 'build']
make_command = ['make', '-C', 'build']

try:
    # Execute the cmake command and print its output
    subprocess.check_call(cmake_command, stderr=subprocess.STDOUT)
    # Execute the make command and print its output
    subprocess.check_call(make_command, stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as e:
    print(e.output)
    sys.exit(e.returncode)


setup(
    name="CLUEstering",
    version=__version__,
    author="Simone Balducci",
    author_email="simone.balducci00@gmail.com",
    description='''A library that generalizes the original 2-dimensional CLUE
                   algorithm made at CERN.''',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['CLUEstering'],
    install_requires=['scikit-learn', 'numpy', 'matplotlib', 'pandas'],
    package_data={'': []},
    keywords=['Python', 'Clustering', 'Binding'],
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Operating System :: Unix',
        'Operating System :: MacOS :: MacOS X'
    ]
)
