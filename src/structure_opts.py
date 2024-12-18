import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import os
import sys

import pickle

from src.coordtrans_phon import get_coordinate_transforms

####################################################################################################
def load_pkl_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_pkl_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

####################################################################################################
### find new structure!!!
# for structure optimization

def get_opt_quantities_from_pkl(file_path):
    data = load_pkl_data(file_path)
    R0, box_lengths, Pmat, num_atoms = data["R0"], data["box_lengths"], data["Pmat"], data["num_atoms"]
    coordinate_type, dim = data["args"].coordinate_type, data["args"].dim

    trans_Q2R_novmap, _ = get_coordinate_transforms(num_atoms, dim, coordinate_type)
    trans_Q2R = jax.vmap(trans_Q2R_novmap, in_axes=(0, None, None, None), out_axes=(0))

    Q = data["x_epoch"]
    acc_steps, num_devices, batch_per_device, num_modes, ndim = Q.shape
    Q = Q.reshape(-1, num_modes, ndim)
    R = trans_Q2R(Q, R0, box_lengths, Pmat)
    box_lengths = np.array(box_lengths)
    
    dpfile = data["args"].dpfile
    return R, box_lengths, num_atoms, dpfile

####################################################################################################
def optimize(dp_forcefn_vmap, 
             coord, 
             box_lengths, 
             steps=50, 
             eta=1e-7
             ):
    for i in range(steps):
        coord = coord + eta * dp_forcefn_vmap(coord, box_lengths)
        coord = coord - box_lengths * jnp.floor(coord / box_lengths)
    return coord


