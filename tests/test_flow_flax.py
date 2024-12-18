import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import time
from jax.flatten_util import ravel_pytree
import numpy as np
import pytest

import os
import sys
sys.path.append("..")
current_dir = os.getcwd()
print("current_dir:", current_dir)
sys.path.append(current_dir)
sys.path.append(current_dir + "/src")

from src.flow_rnvpflax import make_flow_model

####################################################################################################

def test_flow_model():
    key = jax.random.key(42)

    flow_depth = 4
    mlp_width = 32
    mlp_depth = 2
    num_modes = 12
    
    def check_flow_model(flow_st):
        
        print("Testing flow_st: %s" % flow_st)
        
        #========== make flow model with scaling ==========
        flow = make_flow_model(flow_depth, mlp_width, mlp_depth, num_modes, flow_st)
        x = jax.random.uniform(key, (num_modes, 1), dtype=jnp.float64)
        params = flow.init(key, x)
        raveled_params, _ = ravel_pytree(params)
        print("#parameters in the flow model: %d" % raveled_params.size, flush=True)

        t1 = time.time()  
        z, logjacdet_direct = flow.apply(params, x)
        t2 = time.time()
        print("direct logjacdet:", logjacdet_direct, ",  time used:", t2-t1)

        t1 = time.time()
        x_flatten = x.reshape(-1)
        flow_flatten = lambda x: flow.apply(params,x.reshape(num_modes, 1))[0].reshape(-1)
        jac = jax.jacfwd(flow_flatten)(x_flatten)
        _, logjacdet_jacfwd = jnp.linalg.slogdet(jac)
        t2 = time.time()
        print("jacfwd logjacdet:", logjacdet_jacfwd, ",  time used:", t2-t1)
        
        # print("x:", x.shape, x.flatten())
        # print("z:", z.shape, z.flatten())
    
        assert jnp.allclose(logjacdet_direct, logjacdet_jacfwd)
        
    check_flow_model("st")
    check_flow_model("s")
    check_flow_model("t")
    check_flow_model("o")
        
    #assert logjacdet_direct == pytest.approx(logjacdet_jacfwd, rel=1e-9)
    
####################################################################################################
if __name__ == "__main__":
    test_flow_model()
    print("All tests passed!")

'''
cd /home/zhangqi/MLCodes/Vibrational-Solidlithium
python3 tests/test_flow_flax.py
pytest tests/test_flow_flax.py
pytest tests
'''
    