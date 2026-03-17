# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import shutil
import numpy as np
from torch.utils.data import DataLoader
from ase.calculators.calculator import Calculator
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import Dataset, StructureData, get_train_val_test_loader
from chgnet.model.dynamics import CHGNetCalculator as CHGNetCalculatorOriginal

from mlps_finetuning.energy_ref import get_corrected_energy
from mlps_finetuning.utilities import RedirectOutput

# -------------------------------------------------------------------------------------
# CHGNET CALCULATOR
# -------------------------------------------------------------------------------------

class CHGNetCalculator(CHGNetCalculatorOriginal):

    def __init__(
        self,
        model: object = None,
        model_name: str = None,
        logfile: str = None,
        **kwargs: dict,
    ):
        # Load pretrained model.
        if model is None and model_name is not None:
            model = CHGNet.load(model_name=model_name, **kwargs)
        # Initialize.
        with RedirectOutput(logfile=logfile):
            super().__init__(model=model, **kwargs)
        # Counter to keep track of the number of single-point evaluations.
        self.counter = 0
        self.info = {}
    
    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.counter += 1

# -------------------------------------------------------------------------------------
# ATOMS LIST TO DATASET
# -------------------------------------------------------------------------------------

def atoms_list_to_dataset(
    atoms_list: list,
    energy_corr_dict: dict = None,
    targets: str = "efms",
) -> Dataset:
    """
    Convert list of ase Atoms objects into StructureData dataset.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    structure_list = []
    energy_list = []
    forces_list = []
    stress_list = []
    magmoms_list = []
    adaptor = AseAtomsAdaptor()
    for atoms in atoms_list:
        if atoms.calc is None:
            print("No calculator attached to atoms.")
            continue
        if not atoms.calc.results:
            print("No results in calculator.")
            continue
        # Get structure.
        structure = adaptor.get_structure(atoms)
        # Get energy.
        if "energy" in atoms.calc.results:
            # Apply energy correction.
            energy_tot = get_corrected_energy(
                atoms=atoms,
                energy_corr_dict=energy_corr_dict,
            )
            # Get energy per atom.
            energy = energy_tot / len(atoms)
        elif "e" in targets:
            print("Missing energy in results.")
            continue
        else:
            energy = 0.
        # Get forces.
        if "forces" in atoms.calc.results:
            forces = atoms.get_forces(apply_constraint=False)
        elif "f" in targets:
            print("Missing forces in results.")
            continue
        else:
            forces = np.zeros([len(atoms), 3])
        # Get stress.
        if "stress" in atoms.calc.results:
            stress = atoms.get_stress(voigt=False)
        elif "s" in targets:
            print("Missing stress in results.")
            continue
        else:
            stress = np.zeros([3, 3])
        # Get magmoms.
        if "magmoms" in atoms.calc.results:
            magmoms = atoms.get_magnetic_moments()
        elif "m" in targets:
            print("Missing magmoms in results.")
            continue
        else:
            magmoms = np.zeros(len(atoms))
        # Add data to lists.
        structure_list.append(structure)
        energy_list.append(energy)
        forces_list.append(forces)
        stress_list.append(stress)
        magmoms_list.append(magmoms)
    # Build StructureData dataset.
    dataset = StructureData(
        structures=structure_list,
        energies=energy_list,
        forces=forces_list,
        stresses=stress_list if "s" in targets else None,
        magmoms=magmoms_list if "m" in targets else None,
    )
    return dataset

# -------------------------------------------------------------------------------------
# TRAIN CHGNET MODEL
# -------------------------------------------------------------------------------------

def train_CHGNet_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader = None,
    model: object = None,
    model_name: str = "0.3.0",
    targets: str = "efsm",
    optimizer: str = "Adam",
    scheduler: str = "CosLR",
    criterion: str = "MSE",
    epochs: int = 100,
    learning_rate: float = 1e-4,
    use_device: str = None,
    print_freq: int = 10,
    wandb_path: str = None,
    save_dir: str = None,
    save_test_result: bool = False,
    train_composition_model: bool = False,
    logfile: str = None,
    **kwargs: dict,
) -> Trainer:
    """
    Train CHGNet model.
    """
    # Load pretrained CHGNet model.
    if model is None and model_name is not None:
        model = CHGNet.load(model_name=model_name)
    # Define Trainer.
    trainer = Trainer(
        model=model,
        targets=targets,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=epochs,
        learning_rate=learning_rate,
        use_device=use_device,
        print_freq=print_freq,
        wandb_path=wandb_path,
        **kwargs,
    )
    # Train the model.
    with RedirectOutput(logfile=logfile):
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            save_dir=save_dir,
            save_test_result=save_test_result,
            train_composition_model=train_composition_model,
        )
    # Return trainer.
    return trainer

# -------------------------------------------------------------------------------------
# FINETUNE CHGNET MODEL
# -------------------------------------------------------------------------------------

def finetune_CHGNet_model(
    atoms_list: list,
    calc: Calculator = None,
    directory: str = "finetuning",
    label: str = "model",
    energy_corr_dict: dict = None,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    logfile: str = None,
    clean_directory: bool = False,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 4,
    model: object = None,
    model_name: str = "0.3.0",
    targets: str = "efsm",
    optimizer: str = "Adam",
    scheduler: str = "CosLR",
    criterion: str = "MSE",
    use_device: str = None,
    print_freq: int = 10,
    wandb_path: str = None,
    save_dir: str = None,
    save_test_result: bool = False,
    train_composition_model: bool = False,
    calc_kwargs: dict = {},
    **kwargs: dict,
):
    """
    Fine-tune CHGNet model from ase Atoms data.
    """
    # Remove old directory.
    if clean_directory is True and os.path.isdir(directory):
        shutil.rmtree(directory)
    # Start from the model in the calculator.
    if calc is not None:
        model = calc.model
    # Build dataset from atoms_list.
    dataset = atoms_list_to_dataset(
        atoms_list=atoms_list,
        energy_corr_dict=energy_corr_dict,
        targets=targets,
    )
    # Split dataset into training, validation, and test sets.
    train_fraction = 1. - val_fraction - test_fraction
    return_test = True if test_fraction > 0. else False
    loaders = get_train_val_test_loader(
        dataset=dataset,
        batch_size=batch_size,
        train_ratio=train_fraction,
        val_ratio=val_fraction,
        return_test=return_test,
    )
    train_loader, val_loader = loaders[:2]
    test_loader = loaders[2] if return_test else None
    # Run fine-tuning.
    trainer = train_CHGNet_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        model_name=model_name,
        targets=targets,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=epochs,
        learning_rate=learning_rate,
        use_device=use_device,
        print_freq=print_freq,
        wandb_path=wandb_path,
        save_dir=save_dir or os.path.join(directory, label),
        train_composition_model=train_composition_model,
        logfile=logfile,
        **kwargs,
    )
    # Return calculator.
    model = trainer.get_best_model()
    return CHGNetCalculator(model=model, **calc_kwargs)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------