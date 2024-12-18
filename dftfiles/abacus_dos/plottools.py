import numpy as np
import matplotlib.pyplot as plt



####################################################################################################

colors_list = np.array([[100, 149, 237],
                   [8, 81, 156],
                   [214, 39, 40],
                   [139, 0, 0],
                   [50, 205, 50],
                   [34, 139, 34],
                   [255, 165, 0],
                   [255, 127, 80]
                   ],
                )/255

####################################################################################################

def get_colors(colornums = 10, 
               cmap_name = 'jet',
               output_alpha = False
               ):
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(val) for val in np.linspace(0, 1, colornums)]
    if output_alpha:
        colors = np.array(colors)
    else:    
        colors = np.array(colors)[:, 0:3]
    return colors

####################################################################################################

def load_txt_data(file_path, print_type = 1):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    data = np.array(data, dtype=np.float64)
    if print_type == 1:
        print(data.shape)
    return data

####################################################################################################

def wrap_text(text, width):
    return '\n'.join([text[i:i+width] for i in range(0, len(text), width)])

####################################################################################################

