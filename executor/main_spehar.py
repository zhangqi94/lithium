import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import haiku
import flax
import os
import sys
import time
t1 = time.time()

####################################################################################################

def main_spehar(args):
    
    print("Caluculate the harmonic spectra.")

    #========== params from args ==========
    folder = args.folder
    isotope = args.isotope
    dpfile = args.dpfile
    ensemble = args.ensemble
    target_pressure = args.target_pressure
    volume_per_atom = args.volume_per_atom
    supercell_length = jnp.array(args.supercell_length, dtype=jnp.float64)
    supercell_size = jnp.array(args.supercell_size, dtype=jnp.int64)
    lattice_type = args.lattice_type
    coordinate_type = args.coordinate_type
    dim = args.dim
    hessian_type = args.hessian_type
    seed = args.seed

    key = jax.random.key(args.seed)
    
    #========== useful constants ==========
    Kelvin_2_meV = 0.08617333262145
    
    #========== gpu ==========
    print("jax.random.key:", args.seed)
    print("jax.devices:", jax.devices(), flush=True)
    num_devices = jax.device_count()
    print("Number of GPU devices:", num_devices)

####################################################################################################
    print("\n========== Initialize lattice ==========")
    from src.crystal_lithium import get_phys_const, create_supercell, estimated_volume_from_pressure
    h2_over_2m, effective_mass = get_phys_const(isotope)

    if ensemble == "nvt":
        if volume_per_atom < 0:
            raise ValueError("volume_per_atom must be positive for nvt ensemble!!!")
        print("ensemble: nvt (canonical ensemble)") 
        print("volume per atom: %.6f (A^3)" % volume_per_atom)
        R0, box_lengths, num_atoms, shift_vectors, atom_per_unitcell \
                = create_supercell(lattice_type, volume_per_atom, supercell_size, supercell_length)

    elif ensemble == "npt":
        if volume_per_atom < 0:
            volume_per_atom = estimated_volume_from_pressure(target_pressure)
        R0, box_lengths, num_atoms, shift_vectors, atom_per_unitcell \
                = create_supercell(lattice_type, volume_per_atom, supercell_size, supercell_length)
        print("ensemble: npt (isothermal-isobaric ensemble)") 
        if supercell_length is not None:
            print("input initial supercell length:", supercell_length, "(A)")
        print("estimated volume per atom: %.6f (A^3)" % volume_per_atom)
        
    print("isotope:", isotope)
    print("hbar^2/(2m): %.6f (K/A^2)" % h2_over_2m)
    print("effective mass: %.6f" % effective_mass)
    print("lattice type:", lattice_type)
    print("number of atoms: %d" % num_atoms)
    print("atom per unitcell:", atom_per_unitcell)
    print("supercell size:", supercell_size, ", tot:", jnp.prod(supercell_size))
    print("supercell lengths:", box_lengths, "(A)")
    print("supercell volume: %.6f (A^3)" % jnp.prod(box_lengths))
    print("shift vectors: (cell positions & k-point vectors)")
    for ii in range(jnp.prod(supercell_size)):
        print(f"    idx: {ii:3d}    vec: {shift_vectors[ii]}")

####################################################################################################
    print("\n========== Initialize potential energy surface ==========")
    from src.dpjax_lithium import make_dp_model, pkl_mapping, read_pkl_params
    pkl_name = pkl_mapping.get(dpfile)
    dp_energyfn, _ = make_dp_model(pkl_name, num_atoms, box_lengths_init=box_lengths, unit="K")
    Vtot0 = dp_energyfn(R0, box_lengths)
    potential_energy = jax.vmap(dp_energyfn, in_axes=(0, None), out_axes=(0)) 

    print("initialize potential energy of", isotope)
    print("load deepmd model", dpfile, "file from:", pkl_name)
    read_pkl_params(pkl_name, verbosity=2)
    
    print("at equilibrium position: V0 = %.6f (K/atom), %.6f (meV/atom)" 
        %(Vtot0/num_atoms, Vtot0/num_atoms*Kelvin_2_meV), flush=True)

####################################################################################################
    print("\n========== Calculate dynamic matrix ==========")
    from src.coordtrans_phon import get_gradient_and_hessian
    V0, Vgrad, Dmat = get_gradient_and_hessian(R0, box_lengths, effective_mass, 
                                                dp_energyfn, hessian_type=hessian_type)

    print("crystal properties: (in the unit of K^2)")
    print("potential energy V0: %.6f,  per atom: %.6f" %(V0, V0/num_atoms))
    print("V_gradient (first 4 terms):", Vgrad[0:4], "...")
    print("    close to zero (abs<1e-6): %d,    total: %d" 
                            %(jnp.sum(jnp.abs(Vgrad)<1e-6),  Vgrad.shape[0]))

    print("V_hessian (dynamic matrix):", Dmat)
    print("    diagional elements:", jnp.diag(Dmat)[0:4], "...", flush=True)

####################################################################################################
    print("\n========== Initial coordinate transformations ==========")
    from src.coordtrans_phon import init_coordinate_transformations, get_coordinate_transforms

    wsquare_indices, w_indices, Pmat, wkd_rawdata, num_modes \
                                    = init_coordinate_transformations(coordinate_type, 
                                                                    Dmat, 
                                                                    shift_vectors, 
                                                                    supercell_size, 
                                                                    atom_per_unitcell, 
                                                                    num_atoms, 
                                                                    dim, 
                                                                    cal_kpoint = True, 
                                                                    verbosity = 2
                                                                    )

    print("number of modes:", num_modes)

    print("zero-point energy (in meV/atom):")
    from src.orbitals import get_orbitals_1d, get_orbitals_energy
    state_idx = jnp.ones((num_modes, ), dtype=jnp.int64)
    state_energy = get_orbitals_energy(state_idx, w_indices)
    print("    without V0: %.6f    with V0: %.6f" 
        % ((1/effective_mass) * state_energy /num_atoms * Kelvin_2_meV,
           (1/effective_mass) * (state_energy+V0) /num_atoms * Kelvin_2_meV), flush=True)

####################################################################################################
    print("\n========== Calculate frequencies ==========")
    print("compute the frequencies of each mode (in meV):")

    def print_wkd_rawdata(wkd_rawdata):
        w_indices, wsquare_indices, kpoint_indices, kpoint_vectors, _ = wkd_rawdata
        for ii in range(num_atoms*dim):
            print(f"{ii:3d}    {w_indices[ii] * (1/effective_mass) * Kelvin_2_meV:.6f}    "
                f"{kpoint_indices[ii]:3d}    "
                + "    ".join(f"{x:.2f}" for x in (kpoint_vectors[ii])))

    def write_wkd_rawdata(wkd_rawdata, filename):
        w_indices, wsquare_indices, kpoint_indices, kpoint_vectors, _ = wkd_rawdata
        with open(filename, 'w') as file:
            for ii in range(num_atoms*dim):
                line = (f"{ii:3d}    {w_indices[ii] * (1/effective_mass) * Kelvin_2_meV:.6f}    "
                        f"{kpoint_indices[ii]:3d}    "
                        + "    ".join(f"{x:.2f}" for x in (kpoint_vectors[ii])))
                file.write(line + '\n')

    filename = (folder
                + f"datahar_{dpfile}_{lattice_type}_v{volume_per_atom}"
                + f"_{''.join(str(x) for x in supercell_size)}.txt")

    print_wkd_rawdata(wkd_rawdata)
    write_wkd_rawdata(wkd_rawdata, filename)

    t2 = time.time()
    print("save file to:", filename, flush=True)
    print("total time used:", t2-t1, "seconds", flush=True)


####################################################################################################
    
"""
srun -p titanv --cpus-per-task=6 --gres=gpu:1  --pty /bin/bash
srun -p a100 --cpus-per-task=10 --gres=gpu:A100_80G  --pty /bin/bash
srun -p v100 --cpus-per-task=10 --gres=gpu:1  --pty /bin/bash

cd /home/zhangqi/MLCodes/lithium
python3  main.py  --seed 42 \
        --executor "spehar" \
        --folder "/home/zhangqi/MLCodes/lithium/" \
        --isotope "Li"  --dpfile "dp0"  \
        --ensemble "nvt"  --volume_per_atom 7.2  \
        --lattice_type "oC88"  --coordinate_type "phon"  \
        --hessian_type "for1" \
        --supercell_size 3 2 2

cd /home/zhangqi/MLCodes/lithium
python3  main.py  \
        --executor "spehar" \
        --folder "/home/zhangqi/MLCodes/lithium/" \
        --isotope "Li"  --dpfile "dp0"  \
        --ensemble "nvt"  --volume_per_atom 7.2  \
        --lattice_type "cI16"  --coordinate_type "phon"  \
        --hessian_type "for1" \
        --supercell_size 4 4 4
       
cd /home/zhangqi/MLCodes/lithium 
python3  main.py  \
        --executor "spehar" \
        --folder "/home/zhangqi/MLCodes/lithium/" \
        --isotope "Li"  --dpfile "dp4"  \
        --ensemble "nvt"  --volume_per_atom 19.2  \
        --lattice_type "bcc"  --coordinate_type "phon"  \
        --hessian_type "for1" \
        --supercell_size 8 8 8
        
cd /home/zhangqi/MLCodes/lithium 
python3  main.py  \
        --executor "spehar" \
        --folder "/home/zhangqi/MLCodes/lithium/" \
        --isotope "Li"  --dpfile "dp4"  \
        --ensemble "nvt"  --volume_per_atom 19.2  \
        --lattice_type "fcc"  --coordinate_type "phon"  \
        --hessian_type "for1" \
        --supercell_size 6 6 6

cd /home/zhangqi/MLCodes/lithium
python3  main.py  \
        --executor "spehar" \
        --folder "/home/zhangqi/MLCodes/lithium/" \
        --isotope "Li"  --dpfile "dp0"  \
        --ensemble "nvt"  --volume_per_atom 7.2  \
        --lattice_type "oC40"  --coordinate_type "phon"  \
        --hessian_type "for1"  \
        --supercell_size 3 3 3
        
        
cd /mnt/ssht02home/MLCodes/lithium
python3  main.py  \
        --executor "spehar" \
        --folder "/mnt/ssht02home/MLCodes/lithium/" \
        --isotope "Li"  --dpfile "dp4"  \
        --ensemble "nvt"  --volume_per_atom 19.2  \
        --lattice_type "fcc"  --coordinate_type "phon"  \
        --hessian_type "for1"  \
        --supercell_size 4 4 4
        
cd /mnt/ssht02home/MLCodes/lithium
python3  main.py  \
        --executor "spehar" \
        --folder "/mnt/ssht02home/MLCodes/lithium/" \
        --isotope "Li"  --dpfile "dp4"  \
        --ensemble "nvt"  --volume_per_atom 19.2  \
        --lattice_type "bcc"  --coordinate_type "phon"  \
        --hessian_type "for1"  \
        --supercell_size 5 5 5

""" 