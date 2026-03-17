# MLPs fine-tuning

Tools for the **fine-tuning of machine-learning potentials (MLPs)** for atomistic simulations. This repository provides utilities to fine-tune universal MLPs using datasets generated from **density-functional theory (DFT)** calculations. The goal is to enable efficient refinement of universal MLPs for specific materials systems and reaction environments. The package is designed to integrate naturally with workflows based on the **Atomic Simulation Environment (ASE)** and atomistic simulation codes.

## Installation

Clone and install the repository:
```bash
git clone https://github.com/raffaelecheula/mlps_finetuning.git
cd mlps_finetuning
pip install -e .
```

## Compatibility

**Supported MLPs**
- CHGNet
- MACE
- OCP (FairchemV1)
- FAIRChem

**Supported DFT codes**
- Quantum Espresso
- VASP

## Contributing

Contributions to `mlps_finetuning` are welcome. If you have suggestions, bug reports, or would like to contribute code, please open an issue or submit a pull request on the [GitHub repository](https://github.com/raffaelecheula/mlps_finetuning).

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](https://github.com/raffaelecheula/mlps_finetuning/LICENSE) file for details.

## Author

Raffaele Cheula (email: cheula.raffaele@gmail.com)