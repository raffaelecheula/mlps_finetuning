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
        if atoms.calc is None:
            continue
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
    targets,
    optimizer,
    scheduler,
    criterion,
    epochs,
    learning_rate,
    use_device,
    print_freq,
    wandb_path,
    train_loader,
    val_loader,
    test_loader,
    save_dir=None,
    train_composition_model=False,
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
    
    return trainer

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
    return_test=True,
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
    finetune_chgnet(
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

# -------------------------------------------------------------------------------------
# GET TRAIN VAL TEST LOADER FROM INDICES
# -------------------------------------------------------------------------------------

def get_train_val_test_loader_from_indices(
    dataset,
    indices_train,
    indices_val,
    indices_test=[],
    batch_size=8,
    return_test=True,
    num_workers=0,
    pin_memory=True,
):
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
# FINETUNE CHGNET
# -------------------------------------------------------------------------------------

def finetune_chgnet_crossval(
    atoms_list,
    energy_corr_dict=None,
    targets="efsm",
    batch_size=8,
    n_splits=5,
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
    import random
    from sklearn.model_selection import KFold
    # Build dataset from atoms_list.
    dataset = atoms_list_to_dataset(
        atoms_list=atoms_list,
        energy_corr_dict=energy_corr_dict,
        targets=targets,
    )
    indices = list(range(len(dataset)))
    random.shuffle(indices) # is it necessary?
    #TODO: get indices of different cv splits.
    # Probably we can use KFold from sklearn.model_selection but it has to be tested.
    # I don't know if it is a good idea to use a test set or only evaluate in val.
    kfold = KFold(n_splits=n_splits)
    for indices_train, indices_val in kfold.split(indices):
        # Split dataset into training and validation.
        train_loader, val_loader = get_train_val_test_loader_from_indices(
            dataset=dataset,
            batch_size=batch_size,
            indices_train=indices_train,
            indices_val=indices_val,
            return_test=False,
        )
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
            test_loader=None,
            save_dir=save_dir,
            train_composition_model=train_composition_model,
        )
        # TODO: get the results and average them.

# -------------------------------------------------------------------------------------
# FINETUNE CHGNET GROUPS
# -------------------------------------------------------------------------------------

def finetune_chgnet_groups(
    atoms_list,
    groups_name,
    n_splits=5,
    energy_corr_dict=None,
    targets="efsm",
    batch_size=8,
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
    from sklearn.model_selection import GroupKFold
    # Build dataset from atoms_list.
    dataset = atoms_list_to_dataset(
        atoms_list=atoms_list,
        energy_corr_dict=energy_corr_dict,
        targets=targets,
    )
    indices = list(range(len(dataset)))
    # TODO: calculate indices_train and indices_val according to the 
    # atoms.info dictionaries.
    # Probably we can use sklearn.model_selection.GroupKFold
    # Please test if this works.
    groups = [atoms.info[groups_name] for atoms in atoms_list]
    kfold = GroupKFold(n_splits=n_splits)
    for indices_train, indices_val in kfold.split(indices, groups=groups):
        # Split dataset into training and validation.
        train_loader, val_loader = get_train_val_test_loader_from_indices(
            dataset=dataset,
            indices_train=indices_train,
            indices_val=indices_val,
            batch_size=batch_size,
            return_test=False,
        )
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
            test_loader=None,
            save_dir=save_dir,
            train_composition_model=train_composition_model,
        )
        # TODO: get the results and average them.

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------