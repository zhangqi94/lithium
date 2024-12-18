import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn

####################################################################################################
class RealNVP(nn.Module):
    """
        Real-valued non-volume preserving (real NVP) transform.
        The implementation follows the paper "arXiv:1605.08803."
    """
    maskflow: list
    flow_depth: int
    mlp_width: int
    mlp_depth: int
    event_size: int
    flow_st: str

    def setup(self):
        # MLP (Multi-Layer Perceptron) layers for the real NVP.
        self.mlp = [self.build_mlp(self.mlp_width, self.mlp_depth, self.event_size)
                        for _ in range(self.flow_depth)]
        self.zoom = self.param('zoom', nn.initializers.ones, (self.event_size, )
                               )
        
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

    def build_mlp(self, mlp_width, mlp_depth, event_size):
        layers = []
        for _ in range(mlp_depth):
            layers.append(nn.Dense((mlp_width), dtype=jnp.float64))
            layers.append(nn.tanh)
        layers.append(nn.Dense(event_size * 2, 
                                kernel_init=nn.initializers.truncated_normal(stddev=0.0001), 
                                bias_init=nn.initializers.zeros,
                                dtype=jnp.float64))
        return nn.Sequential(layers)

    def coupling_forward(self, x1, x2, l):
        # get shift and log(scale) from x1
        shift_and_logscale = self.mlp[l](x1)
        shift, logscale = jnp.split(shift_and_logscale, 2, axis=-1)
        logscale = jnp.where(self.maskflow[l], 0, jnp.tanh(logscale) * self.zoom)
        
        # transform: y2 = x2 * scale + shift
        y2 = x2 * jnp.exp(logscale) + shift
        # calculate: logjacdet for each layer
        sum_logscale = jnp.sum(logscale)
        return y2, sum_logscale

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
        
        for l in range(self.flow_depth):
            # split x into two parts: x1, x2
            x1 = jnp.where(self.maskflow[l], x_flatten, 0)
            x2 = jnp.where(self.maskflow[l], 0, x_flatten)
            
            # get y2 from fc(x1), and calculate logjacdet = sum_l log(scale_l)
            y2, sum_logscale = self.coupling_forward(x1, x2, l)
            logjacdet += sum_logscale

            # update: [x1, x2] -> [x1, y2]
            x_flatten = jnp.where(self.maskflow[l], x_flatten, y2)
            
        x = jnp.reshape(x_flatten, (d1, d2))
        
        return x, logjacdet 

####################################################################################
def make_maskflow(flow_depth, event_size):
    mask1 = jnp.arange(0, jnp.prod(event_size)) % 2 == 0
    mask1 = (jnp.reshape(mask1, event_size)).astype(bool)

    mask2 = jnp.arange(0, jnp.prod(event_size)) % 2 == 1
    mask2 = (jnp.reshape(mask2, event_size)).astype(bool)

    maskflow = []
    for i in range(flow_depth):
        if i % 2 == 0:
            mask = mask1
        else:   
            mask = mask2
        maskflow += [mask]
    return maskflow

####################################################################################################
def make_flow_model(flow_depth, mlp_width, mlp_depth, num_modes, flow_st=True):

    event_size = num_modes
    maskflow = make_maskflow(flow_depth, event_size)
    flow = RealNVP(maskflow, 
                   flow_depth, 
                   mlp_width, 
                   mlp_depth, 
                   event_size, 
                   flow_st,
                   )
    
    return flow

####################################################################################################
if __name__ == "__main__":
    print("\n========== Test Real NVP ==========")
    import time
    from jax.flatten_util import ravel_pytree
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
        z, logjacdet = flow.apply(params, x)
        t2 = time.time()
        print("direct logjacdet:", logjacdet, ",  time used:", t2-t1)

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
    
## python3 /mnt/ssht02home/t02code/lithium/src/flow_rnvpflax.py
