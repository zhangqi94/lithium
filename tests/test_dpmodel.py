import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import json
import pickle

import os
import sys
sys.path.append("..")
current_dir = os.getcwd()
print("current_dir:", current_dir)
sys.path.append(current_dir)
sys.path.append(current_dir + "/src")

from src.dpjax_lithium import make_dp_model

####################################################################################################

# def generic_test_symmetry(dp_energyfn, num_atoms, dim, L):
#     x = jnp.array( np.random.uniform(0., L, (num_atoms, dim)) )
#     print("x.shape:", x.shape)
#     e = dp_energyfn(x)
#     print("e:", e)

#     print("---- Test the dp model is well-defined under lattice translations of PBC ----")
#     image = np.random.randint(-5, 6, size=(num_atoms, dim)) * L
#     e_image = dp_energyfn(x + image)
#     print("e_image:", e_image)
#     assert jnp.allclose(e_image, e)

#     print("---- Test translation invariance ----")
#     shift = np.random.randn(dim)
#     e_shift = dp_energyfn(x + shift)
#     print("e_shift:", e_shift)
#     assert jnp.allclose(e_shift, e)

#     print("---- Test permutation invariance ----")
#     P = np.random.permutation(num_atoms)
#     e_P = dp_energyfn(x[P, :])
#     print("e_permute:", e_P)
#     assert jnp.allclose(e_P, e)


def test_dp_symmetry():
    num_atoms = 32
    dim, L = 3, 10
    box_lengths = jnp.array([L, L, L])
    pkl_name = "./src/dpmodel/fznp_dp0.pkl"
    dp_energyfn, dp_forcefn = make_dp_model(pkl_name, 
                                            num_atoms, 
                                            box_lengths_init=box_lengths, 
                                            unit="eV")
    def energyfn(coord):
        return dp_energyfn(coord, box_lengths)
    energyfn = jax.jit(energyfn)
    
    x = jnp.array( np.random.uniform(0., L, (num_atoms, dim)) )
    print("x.shape:", x.shape)
    e = energyfn(x)
    print("e:", e)

    print("---- Test the dp model is well-defined under lattice translations of PBC ----")
    image = np.random.randint(-5, 6, size=(num_atoms, dim)) * L
    e_image = energyfn(x + image)
    print("e_image:", e_image)
    assert jnp.allclose(e_image, e)

    print("---- Test translation invariance ----")
    shift = np.random.randn(dim)
    e_shift = energyfn(x + shift)
    print("e_shift:", e_shift)
    assert jnp.allclose(e_shift, e)

    print("---- Test permutation invariance ----")
    P = np.random.permutation(num_atoms)
    e_P = energyfn(x[P, :])
    print("e_permute:", e_P)
    assert jnp.allclose(e_P, e)


def test_dp_jax():
    batch_size = 64
    num_atoms = 32
    dim, L = 3, 10
    box_lengths = jnp.array([L, L, L])
    pkl_name = "./src/dpmodel/fznp_dp0.pkl"
    dp_energyfn, dp_forcefn = make_dp_model(pkl_name, 
                                            num_atoms, 
                                            box_lengths_init=box_lengths, 
                                            unit="eV")
    
    
    def energyfn(coord):
        return dp_energyfn(coord, box_lengths)
    def forcefn(coord):
        return dp_forcefn(coord, box_lengths)
    
    energyfn = jax.jit(energyfn)
    x = jnp.array( np.random.uniform(0., L, (batch_size, num_atoms, dim)) )

    e = energyfn(x[0])
    assert e.shape == ()

    e = jax.vmap(energyfn)(x)
    assert e.shape == (batch_size,)

    f = jax.grad(energyfn)(x[0])
    assert f.shape == (num_atoms, dim)
    
    
####################################################################################################
if __name__ == "__main__":
    test_dp_symmetry()
    test_dp_jax()
    print("All tests passed!")


'''
cd /home/zhangqi/MLCodes/lithium
cd /mnt/t02home/MLCodes/lithium
python3 tests/test_dpmodel.py
pytest tests/test_dpmodel.py
'''
    
    
    