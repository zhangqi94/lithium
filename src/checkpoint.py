import pickle
import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

####################################################################################################
def ckpt_filename(epoch, path):
    return os.path.join(path, "epoch_%06d.pkl" % epoch)

def save_pkl_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_pkl_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

####################################################################################################

# def load_data(filename):
#     with open(filename, "rb") as f:
#         data = pickle.load(f)
#     return data

# def load_txt(filename):
#     with open(filename, "r") as file:
#         lines = file.readlines()
#     data = [line.strip().split() for line in lines]
#     data = np.array(data, dtype=np.float64)
#     return data

