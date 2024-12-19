# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
from ase import Atoms
from ase.db.core import Database
from ase.io import read
from ase.db import connect

# -------------------------------------------------------------------------------------
# GET PATHS WITH PARENTS
# -------------------------------------------------------------------------------------

def get_paths_with_parents(basedir: str, depth: int) -> list:
    """Get paths with parents, at given depth."""
    cwd = os.getcwd()
    os.chdir(basedir)
    path_list = [[os.getcwd(), []]]
    ii = 0
    paths_with_parents = []
    while ii < len(path_list):
        os.chdir(path_list[ii][0])
        path, parents = path_list[ii]
        if len(parents) < depth:
            for new in [new for new in os.listdir(path) if os.path.isdir(new)]:
                path_list.append([os.path.join(path, new), parents+[new]])
        elif len(parents) == depth:
            paths_with_parents.append(path_list[ii])
        ii += 1
    os.chdir(cwd)
    return paths_with_parents

# -------------------------------------------------------------------------------------
# READ ATOMS LIST
# -------------------------------------------------------------------------------------

def read_atoms_list(
    filepath: str,
    index: int,
    read_fun: callable = lambda filepath, index: read(filepath, index=index),
) -> list:
    """Read ase Atoms from file and return a list."""
    atoms_list = read_fun(filepath, index=index)
    atoms_list = atoms_list if isinstance(atoms_list, list) else [atoms_list]
    return atoms_list

# -------------------------------------------------------------------------------------
# STORE INDEX IN INFO
# -------------------------------------------------------------------------------------

def store_index_in_info(atoms_list: list):
    """Store index in atoms.info dictionary."""
    for ii, atoms in enumerate(atoms_list):
        atoms.info["index"] = ii
        atoms.info["relaxed"] = False
    atoms_list[-1].info["relaxed"] = True

# -------------------------------------------------------------------------------------
# STORE UID IN INFO
# -------------------------------------------------------------------------------------

def store_uid_in_info(atoms_list: list):
    """Store id in atoms.info dictionary."""
    import secrets
    identifier = secrets.token_urlsafe(8)
    for atoms in atoms_list:
        atoms.info["uid"] = identifier

# -------------------------------------------------------------------------------------
# GET ATOMS FROM NESTED DIRS
# -------------------------------------------------------------------------------------

def get_atoms_from_nested_dirs(
    basedir: str,
    tree_keys: list,
    filename: str,
    add_info: dict = {},
    index: int = ":",
    read_fun: callable = None,
    store_index: bool = True,
    store_uid: bool = True,
) -> list:
    """Get list of ase Atoms structures from nested directories.
    Works with a path tree structured as: basedir/arg[0]/.../arg[N]/filename.
    The atoms.info will have {tree_keys[0]: arg[0], ..., tree_keys[N]: arg[N]}.
    """
    if store_index is None:
        store_index = True if index == ":" else False
    paths_with_parents = get_paths_with_parents(
        basedir=basedir,
        depth=len(tree_keys),
    )
    atoms_all = []
    for path, parents in paths_with_parents:
        filepath = os.path.join(basedir, path, filename)
        if os.path.isfile(filepath):
            atoms_list = read_atoms_list(
                filepath=filepath,
                index=index,
                read_fun=read_fun,
            )
            if store_index is True:
                store_index_in_info(atoms_list=atoms_list)
            if store_uid is True:
                store_uid_in_info(atoms_list=atoms_list)
            for atoms in atoms_list:
                for kk, key in enumerate(tree_keys):
                    if key is not None:
                        atoms.info[key] = parents[kk]
                atoms.info.update(add_info)
            atoms_all += atoms_list
    return atoms_all

# -------------------------------------------------------------------------------------
# GET ATOMS FROM OS WALK
# -------------------------------------------------------------------------------------

def get_atoms_from_os_walk(
    basedirs: list,
    filename: str,
    index: int = ":",
    read_fun: callable = None,
    store_index: bool = True,
) -> list:
    """Get list of ase Atoms structures from nested directories with os walk."""
    atoms_all = []
    for basedir in basedirs:
        for path, folders, files in os.walk(basedir):
            if filename in files:
                filepath = os.path.join(path, filename)
                atoms_list = read_atoms_list(
                    filepath=filepath,
                    index=index,
                    read_fun=read_fun,
                )
                if store_index is True:
                    store_index_in_info(atoms_list)
                atoms_all += atoms_list
    return atoms_all

# -------------------------------------------------------------------------------------
# WRITE ATOMS LIST TO DB
# -------------------------------------------------------------------------------------

def write_atoms_list_to_db(
    atoms_list: list,
    db_ase: Database,
    keys_store: list = [],
    keys_match: list = None,
    fill_stress: bool = False,
    fill_magmom: bool = False,
):
    """Write list of ase Atoms to ase database."""
    for atoms in atoms_list:
        write_atoms_to_db(
            atoms=atoms,
            db_ase=db_ase,
            keys_store=keys_store,
            keys_match=keys_match,
            fill_stress=fill_stress,
            fill_magmom=fill_magmom,
        )

# -------------------------------------------------------------------------------------
# WRITE ATOMS TO DB
# -------------------------------------------------------------------------------------

def write_atoms_to_db(
    atoms: Atoms,
    db_ase: Database,
    keys_store: list = [],
    keys_match: list = None,
    fill_stress: bool = False,
    fill_magmom: bool = False,
):
    """Write atoms to ase database."""
    # Fill with zeros stress and magmoms.
    if fill_stress and "stress" not in atoms.calc.results:
        atoms.calc.results["stress"] = np.zeros(6)
    if fill_magmom and "magmoms" not in atoms.calc.results:
        atoms.calc.results["magmoms"] = np.zeros(len(atoms))
    # Get dictionary to store atoms.info into the columns of the db.
    kwargs_store = {key: atoms.info[key] for key in keys_store}
    # Get dictionary to check if structure is already in db.
    if keys_match is not None:
        kwargs_match = {key: atoms.info[key] for key in keys_match}
    # Write structure to db.
    if keys_match is None or db_ase.count(**kwargs_match) == 0:
        db_ase.write(atoms=atoms, data=atoms.info, **kwargs_store)
    elif db_ase.count(**kwargs_match) == 1:
        row_id = db_ase.get(**kwargs_match).id
        db_ase.update(id=row_id, atoms=atoms, data=atoms.info, **kwargs_store)

# -------------------------------------------------------------------------------------
# GET ATOMS LIST FROM DB
# -------------------------------------------------------------------------------------

def get_atoms_list_from_db(
    db_ase: Database,
    selection: str = "",
    **kwargs,
) -> list:
    """Get list of ase Atoms from ase database."""
    atoms_list = []
    for id in [aa.id for aa in db_ase.select(selection=selection, **kwargs)]:
        atoms_row = db_ase.get(id=id)
        atoms = atoms_row.toatoms()
        atoms.info = atoms_row.data
        atoms.info.update(atoms_row.key_value_pairs)
        atoms_list.append(atoms)
    return atoms_list

# -------------------------------------------------------------------------------------
# GET ATOMS FROM DB
# -------------------------------------------------------------------------------------

def get_atoms_from_db(
    db_ase: Database,
    selection: str = "",
    none_ok: bool = False,
    **kwargs,
) -> Atoms:
    """Get ase Atoms from ase database."""
    atoms_list = get_atoms_list_from_db(db_ase=db_ase, selection=selection, **kwargs)
    if none_ok is True and len(atoms_list) < 1:
        return None
    elif len(atoms_list) < 1:
        raise RuntimeError("No atoms structure found in database.")
    elif len(atoms_list) > 1:
        raise RuntimeError("More than one atoms structure found in database.")
    return atoms_list[0]

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------