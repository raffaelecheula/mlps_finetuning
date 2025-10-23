# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import timeit
import logging
import shutil
import numpy as np
from ase import Atoms
from ase.io import read, Trajectory
from ase.db import connect

from mlps_finetuning.utilities import print_title
from mlps_finetuning.databases import write_atoms_to_db
from mlps_finetuning.fairchem import (
    FAIRChemCalculator,
    pretrained_mlip,
    finetune_fairchem,
)

from arkimede.workflow.calculations import run_calculation
from arkimede.utilities import filter_results

# -------------------------------------------------------------------------------------
# GET CALCULATOR DFT
# -------------------------------------------------------------------------------------

def get_calculator_dft(
    atoms: Atoms,
):
    """
    Get the DFT calculator.
    """
    from ase.calculators.espresso import EspressoProfile, Espresso
    from qe_toolkit.io import get_pseudopotentials_names
    # Quantum Espresso parameters.
    kpts = None
    input_data = {
        "calculation": "scf",
        "restart_mode": "from_scratch",
        "outdir": "calc",
        "max_seconds": 100000,
        "ecutwfc": 40.0,
        "ecutrho": 320.0,
        "tprnfor": True,
        "occupations": "smearing",
        "degauss": 0.001,
        "smearing": "mv",
        "electron_maxstep": 1000,
        "conv_thr": 1e-06,
        "mixing_mode": "local-TF",
        "mixing_beta": 0.5,
        "diagonalization": "david",
        "diago_david_ndim": 2,
        "startingwfc": "file",
        "startingpot": "file",
    }
    # Quantum Espresso calculator.
    pseudopotentials = get_pseudopotentials_names()
    pseudo_dir = os.getenv("ESPRESSO_PSEUDO")
    profile = EspressoProfile(command="mpirun pw.x", pseudo_dir=pseudo_dir)
    calc_dft = Espresso(
        profile=profile,
        input_data=input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts,
        directory="espresso",
    )
    return calc_dft

# -------------------------------------------------------------------------------------
# GET CALCULATOR MLP
# -------------------------------------------------------------------------------------

def get_calculator_mlp(
    atoms: Atoms,
    model_name: str,
):
    """
    Get the MLP calculator.
    """
    # FAIRChem calculator.
    predict_unit = pretrained_mlip.get_predict_unit(
        model_name=model_name,
        device="cuda",
        cache_dir="../pretrained_models",
    )
    calc_mlp = FAIRChemCalculator(
        predict_unit=predict_unit,
        task_name="oc20",
    )
    # Return calculator.
    return calc_mlp

# -------------------------------------------------------------------------------------
# GET CALCULATOR FINETUNED
# -------------------------------------------------------------------------------------

def get_calculator_finetuned(
    calc_mlp: object,
    label: str,
    model_name: str,
):
    """
    Get fine-tuned MLP calculator.
    """
    # FAIRChem fine-tuning parameters.
    kwargs_trainer = {
        "directory": "finetuning",
        "base_model_name": model_name,
        "checkpoint_path": "/home/rcheula/PythonLibraries/mlps_finetuning/examples/pretrained_models/models--facebook--UMA/snapshots/abaa274e3612b2cfcc5be2d900ffa2a03cb42ee7/checkpoints/uma-s-1p1.pt",
        "dataset_name": "oc20",
        "regression_tasks": "ef",
        "config_dict": {
            "epochs": None, # 100
            "steps": 100, # None
            "batch_size": 1,
            "lr": 1e-2, # 1e-4,
            "weight_decay": 0, # 1e-3,
            "evaluate_every_n_steps": 100,
            "checkpoint_every_n_steps": 100,
        },
        "energy_coeff": 0,
        "forces_coeff": 1,
        #"stress_coeff": 0,
        "db_train_path": "finetuning/train/dft.aselmdb",
        "db_val_path": "finetuning/val/dft.aselmdb",
    }
    # Fine-tune the FAIRChem MLP model.
    checkpoint_path = finetune_fairchem(**kwargs_trainer, label=label)
    # Get fine-tuned MLP calculator.
    predict_unit = pretrained_mlip.load_predict_unit(
        path=checkpoint_path,
        inference_settings="default",
        device="cuda",
    )
    calc_mlp = FAIRChemCalculator(
        predict_unit=predict_unit,
        task_name="oc20",
    )
    #calc_mlp.from_model_checkpoint(
    #    name_or_path=checkpoint_path,
    #    task_name="oc20",
    #)
    # Return calculator.
    return calc_mlp

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Read atoms object.
    from ase.build import fcc100, molecule, add_adsorbate
    from ase.constraints import FixAtoms
    atoms = fcc100("Cu", (2, 2, 3), vacuum=8, periodic=True)
    add_adsorbate(atoms, molecule("CO"), 2.0, "bridge")
    atoms.pbc = True
    atoms.constraints.append(FixAtoms(mask=[aa.position[2] < 10. for aa in atoms]))
    
    # Calculation settings.
    calculation = "relax"
    max_steps = 1000
    min_steps = 10
    fmax = 0.05
    max_steps_actlearn = 100
    
    # Initialize DFT database.
    os.makedirs("finetuning/train", exist_ok=True)
    db_ase = connect(name="finetuning/train/dft.aselmdb", append=False)
    
    # MLP calculator.
    model_name = "uma-s-1p1" # uma-s-1 | uma-s-1p1 | uma-m-1p1
    calc_mlp = get_calculator_mlp(atoms=atoms, model_name=model_name)
    
    # DFT calculator.
    calc_dft = get_calculator_dft(atoms=atoms)

    # Run the active learning protocol.
    time_mlp = 0.
    time_dft = 0.
    time_tun = 0.
    for ii in range(max_steps_actlearn):
        # Run MLP calculation.
        print_title(f"MLP {calculation} calculation.")
        time_mlp_start = timeit.default_timer()
        run_calculation(
            atoms=atoms,
            calculation=calculation,
            calc=calc_mlp,
            max_steps=max_steps,
            min_steps=min_steps,
            fmax=fmax,
            **atoms.info,
        )
        time_mlp += timeit.default_timer()-time_mlp_start
        # Run DFT single-point calculation.
        print_title("DFT single-point calculation.")
        time_dft_start = timeit.default_timer()
        run_calculation(
            atoms=atoms,
            calculation="singlepoint",
            calc=calc_dft,
            no_constraints=True,
            **atoms.info,
        )
        time_dft += timeit.default_timer()-time_dft_start
        # Check DFT max force.
        forces = atoms.get_forces()
        print(forces)
        max_force = np.max(np.linalg.norm(forces, axis=1))
        print(f"DFT fmax = {max_force:+7.4f} [eV/Å]")
        if max_force < fmax:
            break
        # Write structure with DFT forces to db.
        atoms.calc.results["energy"] = calc_mlp.results["energy"]
        atoms.info = {"task": "DFT"}
        write_atoms_to_db(
            atoms=atoms,
            db_ase=db_ase,
            keys_match=["task"],
            keys_store=["task"],
            no_constraints=True,
        )
        natoms = [len(atoms)]
        np.savez_compressed("finetuning/train/metadata.npz", natoms=natoms)
        shutil.copytree("finetuning/train", "finetuning/val", dirs_exist_ok=True)
        # Run MLP model fine-tuning.
        print_title("MLP fine-tuning.")
        time_tun_start = timeit.default_timer()
        label = f"model_{ii+1:02d}"
        calc_mlp = get_calculator_finetuned(
            calc_mlp=calc_mlp,
            label=label,
            model_name=model_name,
        )
        time_tun += timeit.default_timer()-time_tun_start
    # Write atoms trajectory.
    atoms.info.update({
        "DFT_steps": ii+1,
        "time_MLP": time_mlp, 
        "time_DFT": time_dft,
        "time_tun": time_tun,
    })
    with Trajectory(filename=calculation+".traj", mode="a") as traj:
        traj.write(atoms, **filter_results(atoms.calc.results))
    # Calculation finished.
    print_title(f"Active-learning {calculation} calculation finished.")
    for key in ["DFT_steps", "time_MLP", "time_DFT", "time_tun"]:
        print(key, atoms.info[key])

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Output log file.
    logging.basicConfig(filename="log.txt")
    logging.captureWarnings(True)
    # Run script and measure execution time.
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print(f"Execution time = {time_stop-time_start:6.1f} [s].")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
