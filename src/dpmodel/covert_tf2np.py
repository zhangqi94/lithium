from deepmd.calculator import DP
from deepmd.infer import DeepPot
from ase import Atoms
import numpy as np
import tensorflow as tf
import json
import pickle

import os
import sys
# Set the current directory to the script's location
# current_dir = "/home/zhangqi/MLCodes/lithium/src/dpmodel"
current_dir = os.getcwd()
print("current_dir:", current_dir)
sys.path.append(current_dir)

####################################################################################################
# ==== Define the model file and output pickle file names ====
dpmodel_data = "frozen_model_dp0.pb"
dpmodel_pklfile = "fznp_dp0.pkl"

dpmodel_data = "frozen_model_dp2.pb"
dpmodel_pklfile = "fznp_dp2.pkl"

dpmodel_data = "frozen_model_dp3.pb"
dpmodel_pklfile = "fznp_dp3.pkl"

dpmodel_data = "frozen_model_dp4.pb"
dpmodel_pklfile = "fznp_dp4.pkl"

####################################################################################################
print("======== Initialize DeepPot with the frozen model data ========")   
dp = DeepPot(dpmodel_data)
graph_def = dp.graph.as_graph_def()
print("dpmodel data:", dpmodel_data)

print("======== Retrieve the training script from the graph ========")    
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

training_script_value = None
for node in graph_def.node:
    if node.name == "load/train_attr/training_script":
        training_script_value = node.attr["value"].tensor.string_val[0].decode('utf-8')
    
training_script_dict = json.loads(training_script_value)

# Prepare a dictionary to store model metadata
metadata = {"se_e2_a": {}, 
        "se_e3": {}
        }

metadata["se_e2_a"]["nsel"]        = training_script_dict["model"]["descriptor"]["list"][0]["sel"][0]
metadata["se_e2_a"]["rcut_smth"]   = training_script_dict["model"]["descriptor"]["list"][0]["rcut_smth"]
metadata["se_e2_a"]["rcut"]        = training_script_dict["model"]["descriptor"]["list"][0]["rcut"]
metadata["se_e2_a"]["axis_neuron"] = training_script_dict["model"]["descriptor"]["list"][0]["axis_neuron"]

metadata["se_e3"]["nsel"]      = training_script_dict["model"]["descriptor"]["list"][1]["sel"][0]
metadata["se_e3"]["rcut_smth"] = training_script_dict["model"]["descriptor"]["list"][1]["rcut_smth"]
metadata["se_e3"]["rcut"]      = training_script_dict["model"]["descriptor"]["list"][1]["rcut"]
metadata["se_e3"]["ng"]        = training_script_dict["model"]["descriptor"]["list"][1]["neuron"][-1]

print("metadata:", metadata)


print("======== Retrieve training parameters from the graph ========")    
# Initialize a dictionary to store the training parameters
params = {"se_e2_a": [[None for _ in range(2)] for _ in range(3)], 
          "se_e3": [[None for _ in range(2)] for _ in range(3)], 
          "fit": [[None for _ in range(2)] for _ in range(3)], 
          "fit_idt": [None for _ in range(2)], 
          "fit_final": [None for _ in range(2)],
          "em2_avgstd": [None for _ in range(2)],
          "em3_avgstd": [None for _ in range(2)],
          }

# Define a mapping between node names and the corresponding parameters
node_mapping = {
    # se_e2_a mappings
    "load/filter_type_0_0/matrix_1_0": ("se_e2_a", 0, 0),
    "load/filter_type_0_0/bias_1_0": ("se_e2_a", 0, 1),
    "load/filter_type_0_0/matrix_2_0": ("se_e2_a", 1, 0),
    "load/filter_type_0_0/bias_2_0": ("se_e2_a", 1, 1),
    "load/filter_type_0_0/matrix_3_0": ("se_e2_a", 2, 0),
    "load/filter_type_0_0/bias_3_0": ("se_e2_a", 2, 1),
    
    # se_e3 mappings
    "load/filter_type_all_1/matrix_1_0_0": ("se_e3", 0, 0),
    "load/filter_type_all_1/bias_1_0_0": ("se_e3", 0, 1),
    "load/filter_type_all_1/matrix_2_0_0": ("se_e3", 1, 0),
    "load/filter_type_all_1/bias_2_0_0": ("se_e3", 1, 1),
    "load/filter_type_all_1/matrix_3_0_0": ("se_e3", 2, 0),
    "load/filter_type_all_1/bias_3_0_0": ("se_e3", 2, 1),

    # fit mappings
    "load/layer_0_type_0/matrix": ("fit", 0, 0),
    "load/layer_0_type_0/bias": ("fit", 0, 1),
    "load/layer_1_type_0/matrix": ("fit", 1, 0),
    "load/layer_1_type_0/bias": ("fit", 1, 1),
    "load/layer_2_type_0/matrix": ("fit", 2, 0),
    "load/layer_2_type_0/bias": ("fit", 2, 1),

    # fit_idt mappings
    "load/layer_1_type_0/idt": ("fit_idt", 0),
    "load/layer_2_type_0/idt": ("fit_idt", 1),

    # fit_final mappings
    "load/final_layer_type_0/matrix": ("fit_final", 0),
    "load/final_layer_type_0/bias": ("fit_final", 1),
    
    # em2 avg & std
    "load/descrpt_attr_0/t_avg": ("em2_avgstd", 0),
    "load/descrpt_attr_0/t_std": ("em2_avgstd", 1),
    
    # em3 avg & std
    "load/descrpt_attr_1/t_avg": ("em3_avgstd", 0),
    "load/descrpt_attr_1/t_std": ("em3_avgstd", 1),
}

# Function to load tensor data from the session
def load_tensor_data(sess, node_name, node_mapping, params):
    if node_name in node_mapping:
        mapping = node_mapping[node_name]
        param_name = mapping[0]
        tensor = sess.graph.get_tensor_by_name(node_name + ":0")
        data = sess.run(tensor)
        data = np.array(data, dtype=np.float64)
        if len(mapping) == 3:
            i, j = mapping[1], mapping[2]
            params[param_name][i][j] = data
        elif len(mapping) == 2:
            i = mapping[1]
            params[param_name][i] = data
        print("node_name:", node_name, ",", param_name, data.shape)

# Load and process the tensor data from the graph  
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
    with tf.compat.v1.Session(graph=graph) as sess:
        for node in graph_def.node:
            load_tensor_data(sess, node.name, node_mapping, params)
            
####################################################################################################
## Prepare the dpmodel for use with numpy
dpnp = {"metadata": metadata, "params": params}

with open(dpmodel_pklfile, 'wb') as f:
    pickle.dump(dpnp, f)
print("save as:", dpmodel_pklfile)

####################################################################################################
"""
conda activate deepmd
cd /home/zhangqi/MLCodes/lithium/src/dpmodel
python3 covert_tf2np.py
"""

