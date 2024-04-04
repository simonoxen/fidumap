from setuptools import setup, find_packages

setup(
    name='fidumap',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/simonoxen/fidumap',
    install_requires=[
        # list of packages your project depends on
    ],
    entry_points={
        'console_scripts': [
            'fidumap_register = fidumap.register',
            'fidumap_extract = fidumap.extract',
        ],
    })