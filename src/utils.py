import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pickle
import matplotlib.pyplot as plt

####################################################################################################
'''
    Usefull functions for Vibrational-Solidlithium
'''
####################################################################################################

# shard = jax.pmap(lambda x: x)

# def replicate(pytree, num_devices):
#     dummy_input = jnp.empty(num_devices)
#     return jax.pmap(lambda _: pytree)(dummy_input)


####################################################################################################
## get radius distribution function (RDF) for 3 dimensional 
## example: rmesh, gr = get_gr(x[:, :n//2], x[:, :n//2], L, nums=100)
def get_gr(x, y, L, nums=500): 
    batchsize, n, dim = x.shape[0], x.shape[1], x.shape[2]
    
    i,j = np.triu_indices(n, k=1)
    rij = (np.reshape(x, (-1, n, 1, dim)) - np.reshape(y, (-1, 1, n, dim)))[:,i,j]
    rij = rij - L*np.rint(rij/L)
    dij = np.linalg.norm(rij, axis=-1)  # shape: (batchsize, n*(n-1)/2)
    
    hist, bin_edges = np.histogram(dij.reshape(-1,), range=[0, L/2], bins=nums)
    
    dr = bin_edges[1] - bin_edges[0]
    hist = hist*2/(n * batchsize)

    rmesh = bin_edges[0:-1] + dr/2
    
    h_id = 4/3*np.pi*n/(L**3)* ((rmesh+dr)**3 - rmesh**3 )
    gr = hist/h_id
    return rmesh, gr

####################################################################################################
## get title from file_path
## example: title_str = wrap_text(file_name, 50)
def wrap_text(text, width):
    return '\n'.join([text[i:i+width] for i in range(0, len(text), width)])

####################################################################################################
## get colors for plot
## example: colors = get_colors(colornums = 10)
def get_colors(colornums = 10, output_alpha = False):
    cmap = plt.get_cmap('jet')
    colors = [cmap(val) for val in np.linspace(0, 1, colornums)]
    if output_alpha:
        colors = np.array(colors)
    else:    
        colors = np.array(colors)[:, 0:3]
    return colors


def lighten_color(rgb, factor):
    """
    Lightens an RGB color.
        param rgb: (tuple) Original RGB color
        param factor: (float) Lightening factor, range is 0 to 1, closer to 1 means lighter color
        return: (tuple) Lightened RGB color
    """
    # r, g, b = rgb
    # r = r + (1 - r) * factor
    # g = g + (1 - g) * factor
    # b = b + (1 - b) * factor
    # rgb = (r, g, b)
    rgb = rgb + (1 - rgb) * factor
    return rgb

####################################################################################################
## load txt & pkl files
def load_txt_data(file_path, print_type = 1):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    data = np.array(data, dtype=np.float64)
    if print_type == 1:
        print(data.shape)
    return data

def load_pkl_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

####################################################################################################

def moving_average(data_idx, data_values, data_error, window_size=100):
    df_id = data_idx[window_size-1:]
    df_ma = np.convolve(data_values, np.ones(window_size)/window_size, mode='valid')
    df_er = np.sqrt(np.convolve(data_error**2, np.ones(window_size)/(window_size**2), mode='valid'))
    return df_id, df_ma, df_er

####################################################################################################

def read_str(data_str):
    data_lines = data_str.strip().split('\n')
    data_array = np.array([list(map(float, line.split(','))) for line in data_lines])
    print(data_array.shape)
    return data_array

def convert_str2array(phasediagram_str):
    lines = phasediagram_str.strip().split('\n')
    data = [list(map(float, line.split())) for line in lines]
    data_array = np.array(data, dtype=np.float64)
    print(data_array.shape)
    return data_array