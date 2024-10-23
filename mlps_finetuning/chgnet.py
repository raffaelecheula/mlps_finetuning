# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData, get_train_val_test_loader
from chgnet.model.dynamics import CHGNetCalculator

from mlps_finetuning.energy_ref import get_corrected_energy
from mlps_finetuning.finetuning import get_train_val_test_loader_indices

# -------------------------------------------------------------------------------------
# ATOMS LIST TO DATASET
# -------------------------------------------------------------------------------------

def atoms_list_to_dataset(
    atoms_list,
    energy_corr_dict=None,
    targets="efsm",
):
    """Convert list of ase Atoms objects into StructureData dataset."""
    structure_list = []
    energy_list = []
    forces_list = []
    stress_list = []
    magmoms_list = []
    adaptor = AseAtomsAdaptor()
    for atoms in atoms_list:
        # Get structure.
        structure = adaptor.get_structure(atoms)
        # Get energy.
        if "energy" in atoms.calc.results:
            energy = get_corrected_energy(atoms, energy_corr_dict)/len(atoms)
        elif "e" in targets:
            continue
        else:
            energy = 0.
        # Get forces.
        if "forces" in atoms.calc.results:
            forces = atoms.get_forces()
        elif "f" in targets:
            continue
        else:
            forces = np.zeros([len(atoms), 3])
        # Get stress.
        if "stress" in atoms.calc.results:
            stress = atoms.get_stress(voigt=False)
        elif "s" in targets:
            continue
        else:
            stress = np.zeros([3, 3])
        # Get magmoms.
        if "magmoms" in atoms.calc.results:
            magmoms = atoms.get_magnetic_moments()
        elif "m" in targets:
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
    train_loader,
    val_loader,
    test_loader,
    save_dir,
    train_composition_model,
    targets,
    optimizer,
    scheduler,
    criterion,
    epochs,
    learning_rate,
    use_device,
    print_freq,
    wandb_path,
):

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

# -------------------------------------------------------------------------------------
# FINETUNE CHGNET
# -------------------------------------------------------------------------------------

def finetune_chgnet_train_val(
    atoms_list,
    energy_corr_dict=None,
    targets="efsm",
    batch_size=8,
    train_ratio=0.90,
    val_ratio=0.05,
    optimizer="Adam",
    scheduler="CosLR",
    criterion="MSE",
    epochs=100,
    learning_rate=1e-4,
    use_device="cpu",
    print_freq=10,
    wandb_path="chgnet/finetune",
    save_dir=None,
    train_composition_model=False,
):
    """Finetune CHGNet model from ase Atoms data."""
    # Build dataset from atoms_list.
    dataset = atoms_list_to_dataset(
        atoms_list=atoms_list,
        energy_corr_dict=energy_corr_dict,
        targets=targets,
    )
    # Split dataset into training, validation, and test sets.
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    finetune_chgnet(
        train_loader,
        val_loader,
        test_loader,
        save_dir,
        train_composition_model,
        targets,
        optimizer,
        scheduler,
        criterion,
        epochs,
        learning_rate,
        use_device,
        print_freq,
        wandb_path,
    )

# -------------------------------------------------------------------------------------
# FINETUNE CHGNET
# -------------------------------------------------------------------------------------

def finetune_chgnet_crossval(
    
):
    """Finetune CHGNet model from ase Atoms data."""
    # Build dataset from atoms_list.
    dataset = atoms_list_to_dataset(
        atoms_list=atoms_list,
        energy_corr_dict=energy_corr_dict,
        targets=targets,
    )
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    #TODO: get indices of different cv splits.
    
    
    for ii in range(n_crossval):
        # Split dataset into training, validation, and test sets.
        train_loader, val_loader, test_loader = get_train_val_test_loader_indices(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        finetune_chgnet(
            train_loader,
            val_loader,
            test_loader,
            save_dir,
            train_composition_model,
            targets,
            optimizer,
            scheduler,
            criterion,
            epochs,
            learning_rate,
            use_device,
            print_freq,
            wandb_path,
        )

# -------------------------------------------------------------------------------------
# FINETUNE CHGNET
# -------------------------------------------------------------------------------------

def finetune_chgnet_groups():
    # Split dataset into training, validation, and test sets.
    train_loader, val_loader, test_loader = get_train_val_test_loader_indices(
        dataset=dataset,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    finetune_chgnet(
        train_loader,
        val_loader,
        test_loader,
        save_dir,
        train_composition_model,
        targets,
        optimizer,
        scheduler,
        criterion,
        epochs,
        learning_rate,
        use_device,
        print_freq,
        wandb_path,
    )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------