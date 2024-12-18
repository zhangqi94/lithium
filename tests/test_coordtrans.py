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

# from src.crystal_lithium import create_supercell
from src.dpjax_lithium import make_dp_model
from src.coordtrans_phon import get_gradient_and_hessian
from src.coordtrans_phon import init_coordinate_transformations, get_coordinate_transforms

####################################################################################################

def test_coordinate_transforms():
    num_atoms = 32
    dim, L = 3, 10.0
    box_lengths = jnp.array([L, L, L])
    num_modes = (num_atoms - 1) * dim
    R0 = jnp.array( np.random.uniform(0., L, (num_atoms, dim)))

    pkl_name = "./src/dpmodel/fznp_dp0.pkl"
    dp_energyfn, _ = make_dp_model(pkl_name, 
                                   num_atoms, 
                                   box_lengths_init=box_lengths, 
                                   unit="K"
                                   )

    effective_mass = 1.0
    hessian_type = "for1"
    V0, Vgrad, Dmat = get_gradient_and_hessian(R0, 
                                              box_lengths, 
                                              effective_mass, 
                                              dp_energyfn, 
                                              hessian_type=hessian_type
                                              )

    coordinate_type = "phon"

    wsquare_indices, w_indices, Pmat, wkd_rawdata, num_modes \
        = init_coordinate_transformations(coordinate_type, 
                                          Dmat, 
                                          num_atoms = num_atoms, 
                                          cal_kpoint = False, 
                                          verbosity = 2
                                          )

    trans_Q2R_novmap, trans_R2Q_novmap = get_coordinate_transforms(num_atoms, dim, coordinate_type)
    Q1 = jnp.array( np.random.uniform(-0.1, 0.1, (num_modes, 1)) )
    R1 = trans_Q2R_novmap(Q1, R0, box_lengths, Pmat)
    Q2 = trans_R2Q_novmap(R1, R0, box_lengths, Pmat)
    R2 = trans_Q2R_novmap(Q2, R0, box_lengths, Pmat)
    Q3 = trans_R2Q_novmap(R2, R0, box_lengths, Pmat)
    R3 = trans_Q2R_novmap(Q3, R0, box_lengths, Pmat)

    assert jnp.allclose(R1, R2)
    assert jnp.allclose(Q2, Q3)
    assert jnp.allclose(R1, R3)


# ####################################################################################################

if __name__ == "__main__":
    test_coordinate_transforms()
    print("All tests passed!")

'''
cd /home/zhangqi/MLCodes/lithium
cd /mnt/t02home/MLCodes/lithium
python3 tests/test_coordtrans.py
pytest tests/test_coordtrans.py
'''