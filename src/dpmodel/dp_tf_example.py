import numpy as np
from deepmd.infer import DeepPot

####################################################################################################
print("======== test dp-tf model of lithium ========")    
num_atoms = 32
dim, L = 3, 10
box_lengths = np.array([L, L, L])

np.random.seed(42)
coord = np.array( np.random.uniform(0., L, (num_atoms, dim)) )
print("num_atoms:", num_atoms)
print("box_lengths:", box_lengths)
print("coordinate:", coord)

coord = coord.reshape([1, -1])
cell = np.diag(box_lengths).reshape([1, -1])
atype = np.array([0]*num_atoms)

print("========== frozen_model dp0 ==========")
dp = DeepPot("frozen_model_dp0.pb")
e, f, v = dp.eval(coord, cell, atype)
print("energy per atom:", e.item() / num_atoms)
print("force:", f)
# print("stress:", v)

print("========== frozen_model dp2 ==========")
dp = DeepPot("frozen_model_dp2.pb")
e, f, v = dp.eval(coord, cell, atype)
print("energy per atom:", e.item() / num_atoms)
print("force:", f)
# print("stress:", v)

print("========== frozen_model dp3 ==========")
dp = DeepPot("frozen_model_dp3.pb")
e, f, v = dp.eval(coord, cell, atype)
print("energy per atom:", e.item() / num_atoms)
print("force:", f)
# print("stress:", v)

print("========== frozen_model dp4 ==========")
dp = DeepPot("frozen_model_dp4.pb")
e, f, v = dp.eval(coord, cell, atype)
print("energy per atom:", e.item() / num_atoms)
print("force:", f)
# print("stress:", v)

####################################################################################################
'''
ssh t02
conda activate deepmd
cd /home/zhangqi/MLCodes/lithium/src/dpmodel

cd /mnt/t02home/MLCodes/lithium/src/dpmodel
python3 dp_tf_example.py
'''
