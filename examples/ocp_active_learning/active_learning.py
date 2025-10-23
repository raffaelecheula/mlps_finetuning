# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import timeit
import logging
import numpy as np
from contextlib import redirect_stdout
from ase import Atoms
from ase.io import read, Trajectory
from ase.db import connect

from mlps_finetuning.utilities import print_title
from mlps_finetuning.databases import write_atoms_to_db
from mlps_finetuning.ocp import (
    OCPCalculator,
    update_config_yaml,
    default_delete_keys,
    finetune_ocp,
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
    kpts = (6, 6, 1)
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
):
    """
    Get the MLP calculator.
    """
    # OCP calculator.
    #model_name = "GemNet-OC-S2EFS-OC20+OC22"
    #model_name = "EquiformerV2-153M-S2EF-OC20-All+MD"
    #model_name = "EquiformerV2-31M-S2EF-OC20-All+MD"
    #model_name = "EquiformerV2-lE4-lF100-S2EFS-OC22"
    #model_name = "eSCN-L6-M3-Lay20-S2EF-OC20-All+MD"
    model_name = "eSEN-30M-OAM"
    local_cache = "../pretrained_models"
    with redirect_stdout(open("log.txt", "w")):
        calc_mlp = OCPCalculator(
            model_name=model_name,
            local_cache=local_cache,
            cpu=False,
            seed=42,
        )
    return calc_mlp

# -------------------------------------------------------------------------------------
# GET CALCULATOR FINETUNED
# -------------------------------------------------------------------------------------

def get_calculator_finetuned(
    calc_mlp: object,
    label: str,
    from_pretrained: bool = False,
):
    """
    Get fine-tuned MLP calculator.
    """
    # Get config and checkpoint path from info dictionary.
    if "config_init" not in calc_mlp.info:
        config_init = calc_mlp.config.copy()
        checkpoint_init = calc_mlp.config["checkpoint"]
        checkpoint_path = calc_mlp.config["checkpoint"]
    else:
        config_init = calc_mlp.info["config_init"]
        checkpoint_init = calc_mlp.info["checkpoint_init"]
        if from_pretrained is True:
            checkpoint_path = calc_mlp.info["checkpoint_path"]
        else:
            checkpoint_path = calc_mlp.info["checkpoint_init"]
    # OCP fine-tuning parameters.
    config_dict = config_init.copy()
    config_yaml = "finetuning/config.yaml"
    update_keys = {
        "gpus": 1,
        "amp": False,
        "checkpoint": checkpoint_path,
        "optim.eval_every": 10,
        "optim.max_epochs": 100, # 100
        "optim.lr_initial": 1e-4, # 1e-5
        "optim.batch_size": 1,
        "optim.num_workers": 4,
        "optim.energy_coefficient": 1,
        "optim.force_coefficient": 1, # 100
        "task.primary_metric": "forces_mae",
        "logger": "tensorboard", # wandb
        "dataset.train.src": "finetuning/dft.db",
        "dataset.train.format": "ase_db",
        "dataset.train.a2g_args.r_energy": True,
        "dataset.train.a2g_args.r_forces": True,
        "dataset.val.src": "finetuning/dft.db",
        "dataset.test.format": "ase_db",
        "dataset.val.a2g_args.r_energy": True,
        "dataset.val.a2g_args.r_forces": True,
    }
    # Create a new config.yaml file.
    update_config_yaml(
        config_dict=config_dict,
        config_yaml=config_yaml,
        delete_keys=default_delete_keys(),
        update_keys=update_keys,
    )
    # Fine-tune the OCP MLP model.
    checkpoint_path = finetune_ocp(
        checkpoint_path=checkpoint_path,
        config_yaml=config_yaml,
        label=label,
    )
    # Get the fine-tuned MLP calculator.
    with redirect_stdout(open("log.txt", "a")):
        calc_mlp = OCPCalculator(
            trainer="ocp",
            checkpoint_path=checkpoint_path,
            cpu=False,
            seed=42,
        )
    calc_mlp.info.update({
        "config_init": config_init,
        "checkpoint_init": checkpoint_init,
        "checkpoint_path": checkpoint_path,
    })
    # Return calculator.
    return calc_mlp

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():

    # Read atoms object.
    #atoms = read("initial.traj")
    from ase.build import fcc100, molecule, add_adsorbate
    from ase.constraints import FixAtoms
    atoms = fcc100("Cu", (2, 2, 3), vacuum=8, periodic=True)
    add_adsorbate(atoms, molecule("CO"), 2.0, "bridge")
    atoms.pbc = True
    atoms.info = {}
    atoms.constraints.append(FixAtoms(mask=[aa.position[2] < 10. for aa in atoms]))
    
    # Calculation settings.
    calculation = "relax"
    max_steps = 1000
    min_steps = 10
    fmax_mlp = 0.02
    fmax_dft = 0.05
    kwargs_mlp = {}
    kwargs_dft = {}
    max_steps_actlearn = 100
    
    # Initialize DFT database.
    os.makedirs("finetuning", exist_ok=True)
    db_ase = connect(name="finetuning/dft.db", append=False)
    
    # MLP calculator.
    calc_mlp = get_calculator_mlp(atoms=atoms)
    
    # DFT calculator.
    calc_dft = get_calculator_dft(atoms=atoms)

    # Trajectory.
    trajectory = Trajectory(filename=f"{calculation}.traj", mode="w")

    # Run the active learning protocol.
    time_mlp = 0.
    time_dft = 0.
    time_tun = 0.
    for ii in range(max_steps_actlearn):
        # Run MLP calculation.
        print_title(f"MLP {calculation} calculation.")
        time_mlp_start = timeit.default_timer()
        atoms.info["task"] = "MLP"
        run_calculation(
            atoms=atoms,
            calculation=calculation,
            calc=calc_mlp,
            max_steps=max_steps,
            min_steps=min_steps,
            fmax=fmax_mlp,
            trajectory=trajectory,
            **kwargs_mlp,
        )
        time_mlp += timeit.default_timer() - time_mlp_start
        # Run DFT single-point calculation.
        print_title("DFT single-point calculation.")
        time_dft_start = timeit.default_timer()
        atoms.info["task"] = "DFT"
        run_calculation(
            atoms=atoms,
            calculation="singlepoint",
            calc=calc_dft,
            no_constraints=True,
            **kwargs_dft,
        )
        time_dft += timeit.default_timer() - time_dft_start
        # Check DFT max force.
        forces = atoms.get_forces()
        max_force = np.max(np.linalg.norm(forces, axis=1))
        print(f"DFT fmax = {max_force:+7.4f} [eV/Å]")
        if max_force < fmax_dft:
            break
        # Modify DFT energy.
        if "delta_energy" not in atoms.info:
            atoms.info["delta_energy"] = float(
                calc_dft.results["energy"] - calc_mlp.results["energy"]
            )
        atoms.calc.results["energy"] -= atoms.info["delta_energy"]
        # Write structure with DFT forces to db.
        trajectory.write(atoms=atoms, **filter_results(atoms.calc.results))
        write_atoms_to_db(
            atoms=atoms,
            db_ase=db_ase,
            fill_stress=True,
            keys_match=["task"],
            keys_store=["task"],
            no_constraints=True,
        )
        #natoms = [len(atoms)] * (ii+1)
        natoms = [len(atoms)]
        np.savez_compressed("finetuning/metadata.npz", natoms=natoms)
        # Run MLP model fine-tuning.
        print_title("MLP fine-tuning.")
        time_tun_start = timeit.default_timer()
        label = f"model_{ii+1:02d}"
        calc_mlp = get_calculator_finetuned(calc_mlp=calc_mlp, label=label)
        time_tun += timeit.default_timer() - time_tun_start
    # Write atoms trajectory.
    atoms.info.update({
        "DFT_steps": ii+1,
        "time_MLP": time_mlp, 
        "time_DFT": time_dft,
        "time_tun": time_tun,
    })
    with Trajectory(filename=f"{calculation}_final.traj", mode="w") as trajectory:
        trajectory.write(atoms=atoms, **filter_results(atoms.calc.results))
    # Calculation finished.
    print_title(f"Active-learning {calculation} calculation finished.")
    print("DFT_steps =", atoms.info["DFT_steps"])
    for key in ["time_MLP", "time_DFT", "time_tun"]:
        print(f"{key} = {atoms.info[key]:6.1f} [s]")

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Output log file.
    open("log.txt", mode="w").close()
    logging.basicConfig(filename="log.txt", filemode="a")
    logging.captureWarnings(True)
    # Run script and measure execution time.
    time_start = timeit.default_timer()
    main()
    time_stop = timeit.default_timer()
    print(f"Execution time = {time_stop-time_start:6.1f} [s].")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
