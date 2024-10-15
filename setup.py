from setuptools import setup, find_packages

setup(
    name='M-E-GA2',
    version='1.0.0-b2',
    author='Matt Andrews',
    author_email='Matthew.Andrews2024@gmail.com',
    description='Package includes the M_E_Engine and the M_E_GA_Base together these classes facilitate the development'
                ' of Mutable Encoding enables Genetic Algorithms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ML-flash/M-E-GA',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'xxhash',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
)
