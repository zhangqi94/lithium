import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np

import os
import sys
sys.path.append("..")
current_dir = os.getcwd()
print("current_dir:", current_dir)
sys.path.append(current_dir)
sys.path.append(current_dir + "/src")

from src.crystal_lithium import create_supercell
from src.dpjax_lithium import make_dp_model
from src.quantity import pressurefn, stressfn

####################################################################################################
def test_dp_quantitiy():
    # np.random.seed(42)
    batch = 64
    dim = 3
    lattice_type = "fcc"
    volume_per_atom = 18.0
    supercell_size = [2, 2, 2]
    pkl_name = "./src/dpmodel/fznp_dp0.pkl"
    
    R0, box_lengths, num_atoms, _, _ = create_supercell(lattice_type, 
                                                        volume_per_atom, 
                                                        supercell_size, 
                                                        )
    
    dp_energyfn, _ = make_dp_model(pkl_name, 
                                   num_atoms, 
                                   box_lengths_init=box_lengths, 
                                   unit="K"
                                   )
    
    U = jnp.array( np.random.uniform(0., 1., (batch, num_atoms, dim)))
    R = R0 + U
    L = box_lengths
    coord = R - L * jnp.floor(R/L)
    kinetic_energy = 100.0 * jnp.ones((batch, ))
    print("coordinate:", coord.shape)
    
    pressure = pressurefn(dp_energyfn, coord, box_lengths, kinetic_energy)
    forloop = True
    stress = stressfn(dp_energyfn, coord, box_lengths, kinetic_energy, forloop)
    print("pressure:", pressure.shape, pressure.mean())
    print("stress:", stress.shape, stress.mean(axis=0))
    print("trace(stress):", stress.mean())
    
    assert jnp.allclose(pressure.mean(), stress.mean())




####################################################################################################
if __name__ == "__main__":
    test_dp_quantitiy()
    print("All tests passed!")

'''
cd /home/zhangqi/MLCodes/lithium
cd /mnt/t02home/MLCodes/lithium
python3 tests/test_quantitiy.py
pytest tests/test_quantitiy.py
'''
    