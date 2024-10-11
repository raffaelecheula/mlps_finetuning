import setuptools

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

with open('README.md', 'r') as f:
    readme = f.read()

setuptools.setup(
    name="mlps_finetuning",
    version="0.0.1",
    url="https://github.com/raffaelecheula/mlps_finetuning.git",

    author="Raffaele Cheula",
    author_email="cheula.raffaele@gmail.com",

    description=(
        "Tools for the fine-tuning of machine learning potentials (MLPs)."
    ),
    long_description=readme,
    license='GPL-3.0',

    packages=[
        'mlps_finetuning',
    ],
    package_dir={
        'mlps_finetuning': 'mlps_finetuning'
    },
    install_requires=requirements,
    python_requires='>=3.5, <4',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
    ],
)
