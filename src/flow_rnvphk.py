import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import haiku as hk

####################################################################################################
class RealNVP(hk.Module):
    """
    Real-valued non-volume preserving (real NVP) transform.
    The implementation follows the paper "arXiv:1605.08803."
    """
    def __init__(self, maskflow, flow_depth, mlp_width, mlp_depth, event_size, flow_st):
        super().__init__()
        self.maskflow = maskflow
        self.flow_depth = flow_depth
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.event_size = event_size
        self.flow_st = flow_st

        # Initialize MLP layers
        self.fc_mlp = [hk.nets.MLP([mlp_width]*mlp_depth, activation=jax.nn.tanh, activate_final=True)
                        for _ in range(flow_depth)]
        self.fc_lin = [hk.Linear(event_size * 2,
                        w_init=hk.initializers.TruncatedNormal(stddev=0.0001), b_init=jnp.zeros)
                        for _ in range(flow_depth)]
        self.zoom = hk.get_parameter("zoom", [event_size, ], init=jnp.ones, dtype=jnp.float64)

    ################################################################################################
    def coupling_forward(self, x1, x2, l):
        ## get shift and log(scale) from x1
        shift_and_logscale = self.fc_lin[l](self.fc_mlp[l](x1))
        shift, logscale = jnp.split(shift_and_logscale, 2, axis=-1)
        logscale = jnp.where(self.maskflow[l], 0, jnp.tanh(logscale)*self.zoom)
        
        ## transform: y2 = x2 * scale + shift
        y2 = x2 * jnp.exp(logscale) + shift
        
        ## calculate: logjacdet for each layer
        sum_logscale = jnp.sum(logscale)
        return y2, sum_logscale

    ################################################################################################
    def __call__(self, x):
        d1, d2 = x.shape

        # Flatten input
        x_flatten = x.flatten()
        logjacdet = 0.0

        if self.flow_st == "st":
            factor_s = jnp.ones((self.event_size, ), dtype=jnp.float64) + hk.get_parameter("facs", shape=(self.event_size, ), dtype=jnp.float64, init=hk.initializers.TruncatedNormal(0.0001))
            factor_t = hk.get_parameter("fact", shape=(self.event_size, ), dtype=jnp.float64, init=hk.initializers.TruncatedNormal(0.0001))
            x_flatten = factor_s * x_flatten + factor_t
            logjacdet += jnp.sum(jnp.log(factor_s))
        elif self.flow_st == "s":
            factor_s = jnp.ones((self.event_size, ), dtype=jnp.float64) + hk.get_parameter("facs", shape=(self.event_size, ), dtype=jnp.float64, init=hk.initializers.TruncatedNormal(0.0001))
            x_flatten = factor_s * x_flatten
            logjacdet += jnp.sum(jnp.log(factor_s))
        elif self.flow_st == "t":
            factor_t = hk.get_parameter("fact", shape=(self.event_size, ), dtype=jnp.float64, init=hk.initializers.TruncatedNormal(0.0001))
            x_flatten = x_flatten + factor_t
        elif self.flow_st == "o":
            pass  # No scale or shift

        for l in range(self.flow_depth):
            # Split x into two parts: x1, x2
            x1 = jnp.where(self.maskflow[l], x_flatten, 0)
            x2 = jnp.where(self.maskflow[l], 0, x_flatten)

            # Apply coupling forward
            y2, sum_logscale = self.coupling_forward(x1, x2, l)
            logjacdet += sum_logscale

            # Update: [x1, x2] -> [x1, y2]
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

    def forward_fn(x):
        model = RealNVP(maskflow, flow_depth, mlp_width, mlp_depth, event_size, flow_st)
        return model(x)
    flow = hk.transform(forward_fn)
    
    return flow

####################################################################################################
if __name__ == "__main__":
    print("\n========== Test Real NVP ==========")
    import time
    from jax.flatten_util import ravel_pytree
    key = jax.random.PRNGKey(42)
    
    flow_depth = 4
    mlp_width = 32
    mlp_depth = 2
    num_modes = 12

    #========== make flow model with scaling ==========
    def check_flow_model(flow_st):
        
        print("Testing flow_st: %s" % flow_st)
        
        #========== make flow model with scaling ==========
        flow = make_flow_model(flow_depth, mlp_width, mlp_depth, num_modes, flow_st)
        x = jax.random.uniform(key, (num_modes, 1), dtype=jnp.float64)
        params = flow.init(key, x)
        raveled_params, _ = ravel_pytree(params)
        print("#parameters in the flow model: %d" % raveled_params.size, flush=True)

        t1 = time.time()  
        z, logjacdet = flow.apply(params, None, x)
        t2 = time.time()
        print("direct logjacdet:", logjacdet, ",  time used:", t2-t1)

        t1 = time.time()
        x_flatten = x.reshape(-1)
        flow_flatten = lambda x: flow.apply(params, None, x.reshape(num_modes, 1))[0].reshape(-1)
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
    
## python3 /mnt/ssht02home/t02code/lithium/src/flow_rnvphk.py
