import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pickle
import os
import numpy as np

####################################################################################################

#shard = jax.pmap(lambda x: x)

# def shard(x):
#     return jax.pmap(lambda x: x)(x)
def shard(x):
    return x

def replicate(pytree, num_devices):
    dummy_input = jnp.empty(num_devices)
    return jax.pmap(lambda _: pytree)(dummy_input)

def ckpt_filename(epoch, path):
    return os.path.join(path, "epoch_%06d.pkl" % epoch)

####################################################################################################

def automatic_mcstddev(mc_stddev, accept_rate, target_acc=0.4):
    if accept_rate > (target_acc+0.100):
        mc_stddev *= 1.2
    elif (target_acc+0.025) < accept_rate <= (target_acc+0.100):
        mc_stddev *= 1.05
    elif (target_acc-0.025) < accept_rate <= (target_acc+0.025):
        mc_stddev *= 1.0
    elif (target_acc-0.100) < accept_rate <= (target_acc-0.025):
        mc_stddev *= 0.95
    elif accept_rate <= (target_acc-0.100):
        mc_stddev *= 0.8
    return mc_stddev

####################################################################################################
## covert params to float64
def convert_params_dtype(params, dtype=jnp.float64):
    return jax.tree_map(lambda x: x.astype(dtype), params)

###################################################################################################@
## load crystal structures from files
def load_structures(structure_file):
    with open(structure_file, 'r') as file:
        lines = file.readlines()
    
    ## get cell length
    cell_index = lines.index('1.0\n')
    cell = []
    for i in range(cell_index+1, cell_index+4):
        cell.append(list(map(float, lines[i].split())))
    cell = np.array(cell, dtype=np.float64)
    
    ## get fractional coordinates
    direct_index = lines.index('Direct\n')
    coord = []
    for i in range(direct_index + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            break
        if line:
            coord.append(list(map(float, line.split())))
    coord = np.array(coord, dtype=np.float64)

    return cell, coord


