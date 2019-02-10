import setuptools
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

install_requires = [
                       # 'z5py',
                       # 'scikit-learn',
                       # 'nifty',
                       # 'numpy'
]

setuptools.setup(
    name='pias',
    python_requires='>=3',
    packages=['pias'],
    version='0.1.0.dev',
    author='Philipp Hanslovsky',
    author_email='hanslovskyp@janelia.hhmi.org',
    description='Interactive agglomeration scheme for paintera',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Public domain',
    url='https://github.com/saalfeldlab/pias',
    install_requires=install_requires,
    tests_require=['nose'],
    test_suite = 'nose.collector'
)