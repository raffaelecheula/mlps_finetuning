# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from torch.utils.data import DataLoader
from ase.calculators.calculator import Calculator
from sklearn.model_selection import BaseCrossValidator, KFold
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import Dataset, StructureData, get_train_val_test_loader
from chgnet.model.dynamics import CHGNetCalculator

from mlps_finetuning.energy_ref import get_corrected_energy

# -------------------------------------------------------------------------------------
# ATOMS LIST TO DATASET
# -------------------------------------------------------------------------------------

def atoms_list_to_dataset(
    atoms_list: list,
    energy_corr_dict: dict = None,
    targets: str = "efms",
) -> Dataset:
    """Convert list of ase Atoms objects into StructureData dataset."""
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
            energy = get_corrected_energy(atoms, energy_corr_dict)/len(atoms)
        elif "e" in targets:
            print("Missing energy in results.")
            continue
        else:
            energy = 0.
        # Get forces.
        if "forces" in atoms.calc.results:
            forces = atoms.get_forces()
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
# FINETUNE CHGNET
# -------------------------------------------------------------------------------------

def finetune_chgnet(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader = None,
    targets: str = "efsm",
    optimizer: str = "Adam",
    scheduler: str = "CosLR",
    criterion: str = "MSE",
    epochs: int = 100,
    learning_rate: float = 1e-4,
    use_device: str = None,
    print_freq: int = 10,
    wandb_path: str = "chgnet",
    save_dir: str = None,
    train_composition_model: bool = False,
) -> Trainer:
    """Finetune CHGNet model."""
    # Load pretrained CHGNet model.
    model = CHGNet.load()
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
    )
    # Start training.
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=save_dir,
        train_composition_model=train_composition_model,
    )
    # Return trainer.
    return trainer

# -------------------------------------------------------------------------------------
# FINETUNE CHGNET TRAIN VAL
# -------------------------------------------------------------------------------------

def finetune_chgnet_train_val(
    atoms_list: list,
    energy_corr_dict: dict = None,
    targets: str = "efsm",
    batch_size: int = 8,
    train_ratio: float = 0.90,
    val_ratio: float = 0.10,
    return_test: bool = False,
    optimizer: str = "Adam",
    scheduler: str = "CosLR",
    criterion: str = "MSE",
    epochs: int = 100,
    learning_rate: float = 1e-4,
    use_device: str = None,
    print_freq: int = 10,
    wandb_path: str = "chgnet",
    save_dir: str = None,
    train_composition_model: bool = False,
    return_calculator: bool = True,
):
    """Finetune CHGNet model from ase Atoms data."""
    # Build dataset from atoms_list.
    dataset = atoms_list_to_dataset(
        atoms_list=atoms_list,
        energy_corr_dict=energy_corr_dict,
        targets=targets,
    )
    # Split dataset into training, validation, and test sets.
    loaders = get_train_val_test_loader(
        dataset=dataset,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        return_test=return_test,
    )
    train_loader, val_loader = loaders[:2]
    test_loader = loaders[2] if return_test else None
    # Run finetuning.
    trainer = finetune_chgnet(
        targets=targets,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=epochs,
        learning_rate=learning_rate,
        use_device=use_device,
        print_freq=print_freq,
        wandb_path=wandb_path,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=save_dir,
        train_composition_model=train_composition_model,
    )
    # Return calculator.
    if return_calculator:
        model = trainer.get_best_model()
        return CHGNetCalculator(use_device=use_device, model=model)
    else:
        return trainer

# -------------------------------------------------------------------------------------
# GET TRAIN VAL TEST LOADER FROM INDICES
# -------------------------------------------------------------------------------------

def get_train_val_test_loader_from_indices(
    dataset: Dataset,
    indices_train: list,
    indices_val: list,
    indices_test: list = [],
    batch_size: int = 8,
    return_test: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple:
    """Partition a dataset into train, val, test loaders according to indices."""
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    from chgnet.data.dataset import collate_graphs
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(indices=indices_train),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        sampler=SubsetRandomSampler(indices=indices_val),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            sampler=SubsetRandomSampler(indices=indices_test),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader
    return train_loader, val_loader

# -------------------------------------------------------------------------------------
# FINETUNE CHGNET CROSSVAL
# -------------------------------------------------------------------------------------

def finetune_chgnet_crossval(
    atoms_list: list,
    energy_corr_dict: dict = None,
    key_groups: str = None,
    targets: str = "efsm",
    batch_size: int = 8,
    n_splits: int = 5,
    optimizer: str = "Adam",
    scheduler: str = "CosLR",
    criterion: str = "MSE",
    epochs: int = 100,
    learning_rate: float = 1e-4,
    use_device: str = None,
    print_freq: int = 10,
    wandb_path: str = "chgnet",
    save_dir: str = None,
    train_composition_model: bool = False,
    return_test: bool = False,
    random_state: int = 0,
    crossval: BaseCrossValidator = KFold,
    kwargs_crossval: dict = {"random_state": 42, "shuffle": True},
):
    """Finetune CHGNet model using cross-validation on ASE Atoms data."""
    # Convert atoms_list into a dataset.
    dataset = atoms_list_to_dataset(
        atoms_list=atoms_list,
        energy_corr_dict=energy_corr_dict,
        targets=targets,
    )
    # Check if dataset is large enough for the specified number of splits.
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")
    if dataset_size < n_splits:
        raise ValueError(
            f"Not enough samples ({dataset_size}) for {n_splits} splits."
        )
    # Perform Cross-Validation.
    kfold = crossval(n_splits=n_splits, **kwargs_crossval)
    groups = [atoms.info[key_groups] for atoms in atoms_list] if key_groups else None
    MAE_energy_list = []
    MAE_forces_list = []
    indices = list(range(dataset_size))
    for fold, (indices_train, indices_val) in enumerate(
        kfold.split(indices, groups=groups)
    ):
        # Create data loaders for training and validation sets.
        train_loader, val_loader = get_train_val_test_loader_from_indices(
            dataset=dataset,
            batch_size=batch_size,
            indices_train=indices_train,
            indices_val=indices_val,
            return_test=return_test,
        )
        # Run fine-tuning.
        trainer = finetune_chgnet(
            targets=targets,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epochs=epochs,
            learning_rate=learning_rate,
            use_device=use_device,
            print_freq=print_freq,
            wandb_path=wandb_path,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            save_dir=save_dir,
            train_composition_model=train_composition_model,
        )
        # Collect results for this fold.
        MAE_energy = np.min(trainer.training_history["e"]["val"])
        MAE_forces = np.min(trainer.training_history["f"]["val"])
        MAE_energy_list.append(MAE_energy)
        MAE_forces_list.append(MAE_forces)
    # Calculate and print average results.
    MAE_energy_ave = np.mean(MAE_energy_list)
    MAE_energy_std = np.std(MAE_energy_list)
    MAE_forces_ave = np.mean(MAE_forces_list)
    MAE_forces_std = np.std(MAE_forces_list)
    print(f"MAE Val energy: {MAE_energy_ave:.4f} ± {MAE_energy_std:.4f} eV/atom")
    print(f"MAE Val forces: {MAE_forces_ave:.4f} ± {MAE_forces_std:.4f} eV/Å")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------