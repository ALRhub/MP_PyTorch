from setuptools import setup

setup(
    name='MP_PyTorch',
    version='1.0.0',
    packages=['mp_pytorch', 'mp_pytorch.mp', 'mp_pytorch.util',
              'mp_pytorch.basis_gn', 'mp_pytorch.phase_gn'],
    url='https://github.com/ALRhub/MP_PyTorch',
    license='MIT',
    author='Ge Li @ ALR, KIT',
    author_email='ge.li@kit.edu',
    description='Movement Primitives in PyTorch',
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'addict',
    ]
)
