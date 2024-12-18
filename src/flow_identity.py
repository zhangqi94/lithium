import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn

####################################################################################################
class IdentityFlow(nn.Module):

    event_size: int
    flow_st: str = "st"

    def setup(self):
        # MLP (Multi-Layer Perceptron) layers for the real NVP.
        # self.factor_s = self.param('factor_scale', nn.initializers.ones,  (self.event_size, ))
        # self.factor_t = self.param('factor_shift', nn.initializers.zeros, (self.event_size, ))
        
        factor_scale_initializer = nn.initializers.normal(stddev=0.01)
        if self.flow_st == "st":
            self.factor_s = self.param('factor_scale', 
                                    lambda key, shape: 1.0 + factor_scale_initializer(key, shape), 
                                    (self.event_size, ))
            self.factor_t = self.param('factor_shift', 
                                    lambda key, shape: 0.0 + factor_scale_initializer(key, shape), 
                                    (self.event_size, ))
        elif self.flow_st == "s": # only scale
            self.factor_s = self.param('factor_scale', 
                                    lambda key, shape: 1.0 + factor_scale_initializer(key, shape), 
                                    (self.event_size, ))
        elif self.flow_st == "t": # only shift
            self.factor_t = self.param('factor_shift', 
                                    lambda key, shape: 0.0 + factor_scale_initializer(key, shape), 
                                    (self.event_size, ))
        elif self.flow_st == "o": # off scale and shift
            self.f = self.param('f', nn.initializers.ones,  (1, ))
        else:
            raise ValueError("Invalid flow_st value: %s" % self.flow_st)

    def __call__(self, x):
        # Real NVP (forward)
        # x.shape should be: d1 = num_modes, d2 = 1
        d1, d2 = x.shape  
        
        # initial x and logjacdet
        x_flatten = x.flatten()
        if self.flow_st == "st":
            x_flatten = self.factor_s * x_flatten + self.factor_t
            logjacdet = jnp.sum(jnp.log(self.factor_s))
        elif self.flow_st == "s":
            x_flatten = self.factor_s * x_flatten
            logjacdet = jnp.sum(jnp.log(self.factor_s))
        elif self.flow_st == "t":
            x_flatten = x_flatten + self.factor_t
            logjacdet = 0.0
        elif self.flow_st == "o":
            logjacdet = 0.0
            
        x = jnp.reshape(x_flatten, (d1, d2))
        
        return x, logjacdet 

####################################################################################################
def make_flow_model(num_modes, flow_st):

    event_size = num_modes
    flow = IdentityFlow(event_size, flow_st)
    
    return flow

####################################################################################################
if __name__ == "__main__":
    print("\n========== Test Real NVP ==========")
    import time
    from jax.flatten_util import ravel_pytree
    key = jax.random.key(42)
    
    num_modes = 12
    
    def check_flow_model(flow_st):

        print("Testing flow_st: %s" % flow_st)
        flow = make_flow_model(num_modes, flow_st)
        
        x = jax.random.uniform(key, (num_modes, 1), dtype=jnp.float64)
        params = flow.init(key, x)
        
        raveled_params, _ = ravel_pytree(params)
        print("#parameters in the flow model: %d" % raveled_params.size, flush=True)

        t1 = time.time()  
        z, logjacdet = flow.apply(params, x)
        t2 = time.time()
        print("direct logjacdet:", logjacdet, ",  time used:", t2-t1)
        # print("x:", x.shape, x.flatten())
        # print("z:", z.shape, z.flatten())

        t1 = time.time()
        x_flatten = x.reshape(-1)
        flow_flatten = lambda x: flow.apply(params,x.reshape(num_modes, 1))[0].reshape(-1)
        jac = jax.jacfwd(flow_flatten)(x_flatten)
        _, logjacdet = jnp.linalg.slogdet(jac)
        t2 = time.time()
        print("jacfwd logjacdet:", logjacdet, ",  time used:", t2-t1)
        print("x:", x.shape, x.flatten())
        print("z:", z.shape, z.flatten())

    check_flow_model("st")
    check_flow_model("s")
    check_flow_model("t")
    check_flow_model("o")
    
## python3 /mnt/t02home/MLCodes/lithium/src/flow_identity.py
