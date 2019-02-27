import setuptools
import os

name = 'pias'
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

version_info = {}
with open(os.path.join(here, name, 'version_info.py')) as fp:
    exec(fp.read(), version_info)
version = version_info['_version']


# z5py and nifty not on pypi (and probably will never be). nifty is even wrong package
install_requires = [
    # 'z5py',
    'scikit-learn',
    # 'nifty',
    'numpy',
    'zmq'
]

console_scripts = [
    'pias=pias:solver_server_main'
]

entry_points = dict(console_scripts=console_scripts)

packages = [
    f'{name}',
    f'{name}.ext',
    f'{name}.threading',
    f'{name}.zmq_util'
]

setuptools.setup(
    name='pias',
    python_requires='>=3.7',
    packages=packages,
    version=f'{version}',
    author='Philipp Hanslovsky',
    author_email='hanslovskyp@janelia.hhmi.org',
    description='Interactive agglomeration scheme for paintera',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='Public domain',
    url='https://github.com/saalfeldlab/pias',
    install_requires=install_requires,
    tests_require=['nose'],
    test_suite = 'nose.collector',
    entry_points=entry_points
)