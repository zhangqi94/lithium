import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import os
import sys

from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.diffraction.neutron import NDCalculator
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import pickle

from .coordtrans_phon import get_coordinate_transforms
from .crystal_lithium import create_supercell

####################################################################################################
def load_pkl_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_pkl_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
        
####################################################################################################
def get_quantities_from_pkl(file_path,
                            basemode="unitcell",
                            ):
    data = load_pkl_data(file_path)
    R0, box_lengths, Pmat, num_atoms = data["R0"], data["box_lengths"], data["Pmat"], data["num_atoms"]
    coordinate_type, dim = data["args"].coordinate_type, data["args"].dim

    trans_Q2R_novmap, _ = get_coordinate_transforms(num_atoms, dim, coordinate_type)
    trans_Q2R = jax.vmap(trans_Q2R_novmap, in_axes=(0, None, None, None), out_axes=(0))

    Q = data["x_epoch"]
    acc_steps, num_devices, batch_per_device, num_modes, ndim = Q.shape
    Q = Q.reshape(-1, num_modes, ndim)
    R = trans_Q2R(Q, R0, box_lengths, Pmat)
    L = np.array(box_lengths)
    
    lattice_type = data["args"].lattice_type
    supercell_size = np.array(data["args"].supercell_size)
    batch = data["args"].batch
    
    if basemode=="unitcell":
        L0 = L / supercell_size
        num_atoms_unitcell = num_atoms // np.prod(supercell_size)
        R0 = R0[0:num_atoms_unitcell, :]
        
    elif basemode=="supercell":
        L0 = L

    return R, L, num_atoms, R0, L0, batch

####################################################################################################
## plot spectra with Lorentz function
def lorentz(E, E0, gamma):
    return (1 / np.pi) * (gamma / ((E - E0)**2 + gamma**2))

def get_hist_lorentz_xrd(angles, 
                     intensities, 
                     lx=np.linspace(0, 90, 10000), 
                     gamma=0.1
                     ):
    ly = np.zeros_like(lx)
    for i in range(len(angles)):
        ly += intensities[i] * lorentz(lx, angles[i], gamma)  
    ly = ly
    return lx, ly

####################################################################################################
def get_colors(colornums = 10, output_alpha = False):
    cmap = plt.get_cmap('jet')
    colors = [cmap(val) for val in np.linspace(0, 1, colornums)]
    if output_alpha:
        colors = np.array(colors)
    else:    
        colors = np.array(colors)[:, 0:3]
    return colors

####################################################################################################

def create_unitcell(lattice_type = "bcc", 
                    volume_per_atom = -1,
                    unitcell_length = np.array([-1, -1, -1]), ## L0 in the func: get_quantities_from_pkl
                    ):
    R0, L0, num_atoms, shift_vectors, atom_per_unitcell = \
        create_supercell(lattice_type = lattice_type, 
                         volume_per_atom = volume_per_atom,
                         supercell_size = [1, 1, 1],
                         supercell_length = unitcell_length,
                         structure_path = "../src/structures",
                         )
    return R0, L0


####################################################################################################
def compute_xrd_pattern(i, 
               lattice, 
               species, 
               R, 
               L, 
               xrd_calculator, 
               two_theta_range
               ):
    """Compute XRD pattern for the given configuration index `i`."""
    coords = R[i] / L
    structure = Structure(lattice, species, coords)
    pattern = xrd_calculator.get_pattern(structure, two_theta_range)
    return pattern.x, pattern.y

def calculate_xrd_spectrum(L=None, 
                           R=None, 
                           wavelength=1.54, 
                           two_theta_range=(0, 90), 
                           mode="base",
                           ):
    
    if mode=="base":
        R0, L = np.array(R), np.array(L)
        num_atoms = R0.shape[0]
        xrd_calculator = XRDCalculator(wavelength = wavelength)
        lattice = np.diag(L)
        species = ["Li"] * num_atoms
        coords = R0/L
        structure = Structure(lattice, species, coords)
        pattern = xrd_calculator.get_pattern(structure, two_theta_range)
        pattern_x, pattern_y, pattern_hkls = pattern.x, pattern.y, pattern.hkls
        return pattern_x, pattern_y, pattern_hkls
    
    elif mode=="flow":
        R, L = np.array(R), np.array(L)
        batch, num_atoms, _ = R.shape
        xrd_calculator = XRDCalculator(wavelength = wavelength)
        lattice = np.diag(L)
        species = ["Li"] * num_atoms

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                compute_xrd_pattern, 
                range(batch), 
                [lattice] * batch, 
                [species] * batch, 
                [R] * batch, 
                [L] * batch, 
                [xrd_calculator] * batch, 
                [two_theta_range] * batch,
            ))
            
        pattern_x_list, pattern_y_list = zip(*results)
        pattern_x = np.concatenate(pattern_x_list)
        pattern_y = np.concatenate(pattern_y_list)
        return pattern_x, pattern_y
    
####################################################################################################
def compute_nd_pattern(i, 
               lattice, 
               species, 
               R, 
               L, 
               nd_calculator, 
               two_theta_range
               ):
    """Compute ND pattern for the given configuration index `i`."""
    coords = R[i] / L
    structure = Structure(lattice, species, coords)
    pattern = nd_calculator.get_pattern(structure, two_theta_range)
    return pattern.x, pattern.y

def calculate_nd_spectrum(L=None, 
                          R=None, 
                          wavelength=1.54, 
                          two_theta_range=(0, 90), 
                          mode="base",
                          ):
    
    if mode=="base":
        R0, L = np.array(R), np.array(L)
        num_atoms = R0.shape[0]
        nd_calculator = NDCalculator(wavelength = wavelength)
        lattice = np.diag(L)
        species = ["Li"] * num_atoms
        coords = R0/L
        structure = Structure(lattice, species, coords)
        pattern = nd_calculator.get_pattern(structure, two_theta_range)
        pattern_x, pattern_y, pattern_hkls = pattern.x, pattern.y, pattern.hkls
        return pattern_x, pattern_y, pattern_hkls
    
    elif mode=="flow":
        R, L = np.array(R), np.array(L)
        batch, num_atoms, _ = R.shape
        nd_calculator = NDCalculator(wavelength = wavelength)
        lattice = np.diag(L)
        species = ["Li"] * num_atoms

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                compute_nd_pattern, 
                range(batch), 
                [lattice] * batch, 
                [species] * batch, 
                [R] * batch, 
                [L] * batch, 
                [nd_calculator] * batch, 
                [two_theta_range] * batch,
            ))
            
        pattern_x_list, pattern_y_list = zip(*results)
        pattern_x = np.concatenate(pattern_x_list)
        pattern_y = np.concatenate(pattern_y_list)
        return pattern_x, pattern_y

####################################################################################################

