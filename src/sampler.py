import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import haiku as hk
import itertools

####################################################################################################
def make_autoregressive_sampler(van, 
                                num_levels, 
                                sequence_length, 
                                indices_group, 
                                w_indices, 
                                beta,
                                output_logits = False,
                                ):
    """
    Constructs an autoregressive sampler and log-probability calculator for a given
    variational autoregressive network (VAN).

    Inputs:
        - van (object): The variational autoregressive network (VAN) to generate logits.
        - num_levels (int): The number of discrete levels for each element in the sequence.
        - sequence_length (int): The length of the sequence to be generated.
        - indices_group (int): The size of the index group for generating combinations.
        - w_indices (array): Frequency indices for initializing the sampler.
        - beta (float): The inverse temperature for initializing the sampler.

    Outputs:
        - sampler (function): A function that generates samples from the autoregressive
                                model given parameters and a random key.
        - log_prob (function): A function that computes the log-probability of a given
                                sequence under the autoregressive model.
        - index_list (jnp.array): An array of all possible index combinations.
    """

    #========== get combination indices ==========
    def _generate_combinations(num_levels):
        combinations = list(itertools.product(range(num_levels), repeat=indices_group))
        index_list = jnp.array(sorted(combinations, key=lambda x: np.sum(x)), dtype = jnp.int64)
        return index_list
    index_list = _generate_combinations(num_levels)
    
    group_w = w_indices.reshape(sequence_length, indices_group)
    beta_bands = beta * jnp.einsum("id,jd->ij", group_w, index_list)
    #========== get logits from van ==========
    def _logits(params, state_indices):
        #### logits.shape: (sequence_length, num_levels**indices_group)
        # state_indices_expanded = index_list[state_indices]
        # logits = van.apply(params, None, state_indices_expanded)
        
        logits = van.apply(params, None, state_indices.reshape(-1, 1))
        logits = logits - beta_bands
        return logits
    _logits_vmap = jax.vmap(_logits, in_axes = (None, 0), out_axes = (0))

    #========== make sampler (foriloop version) ==========
    def sampler(params, key, batch):
        
        def body_fun(i, carry):
            state_indices, key = carry
            key, subkey = jax.random.split(key)
            logits = _logits_vmap(params, state_indices)
            state_indices = state_indices.at[:, i].\
                            set(jax.random.categorical(subkey, logits[:, i, :], axis=-1))
            return state_indices, key

        state_indices = jnp.zeros((batch, sequence_length), dtype=jnp.int64)
        state_indices, key = jax.lax.fori_loop(0, sequence_length, body_fun, (state_indices, key))
        return state_indices

    #========== calculate log_prob for states ==========
    def log_prob(params, state_indices):
        logits = _logits(params, state_indices)
        logp = jax.nn.log_softmax(logits, axis=-1)
        state_onehot = jax.nn.one_hot(state_indices, num_levels**indices_group)
        logp = jnp.sum(logp * state_onehot)
        return logp

    if output_logits:
        return sampler, log_prob, index_list, _logits
    else:
        return sampler, log_prob, index_list

####################################################################################################
####################################################################################################
if __name__ == '__main__':

    from autoregressive import make_autoregressive_model
    from jax.flatten_util import ravel_pytree
    
    key = jax.random.key(42) 
    num_levels = 2
    indices_group = 3
    sequence_length = 16
    num_modes = sequence_length * indices_group
    beta = 0.1
    batch = 6
    
    van_layers = 2
    van_size = 16
    van_heads = 8
    van_hidden = 32
    
    # van_layers = 1
    # van_size = 8
    # van_heads = 1
    # van_hidden = 8
    
    van = make_autoregressive_model(num_levels, 
                                    indices_group,
                                    van_layers, 
                                    van_size, 
                                    van_heads, 
                                    van_hidden
                                    )
    
    params = van.init(key, jnp.zeros((sequence_length, 1), dtype=jnp.float64))
    print("autoregressive model  [num_levels: %d,  sampler group: %d]" 
                                % (num_levels, indices_group))
    print("                      [layers: %d,  size: %d,  heads: %d,  hidden: %d]" 
                                %(van_layers, van_size, van_heads, van_hidden))
    print("sequence length: %d,  group levels: %d" %(sequence_length, num_levels**indices_group))
    raveled_params, _ = ravel_pytree(params)
    print("#parameters in the model: %d" % raveled_params.size, flush=True)

    w_indices = jax.random.uniform(key, (num_modes,), minval=1, maxval=2)
    # w_indices = jax.random.uniform(key, (num_modes,), minval=0, maxval=0)
    sampler, log_prob_novmap, index_list, _logits = make_autoregressive_sampler(van, 
                                                                                num_levels, 
                                                                                sequence_length, 
                                                                                indices_group, 
                                                                                w_indices, 
                                                                                beta,
                                                                                output_logits=True
                                                                                )
    
    state_indices = jax.random.randint(key, (sequence_length, ), 0, num_levels**indices_group)

    log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)
    state_indices = sampler(params, key, batch)
    print("state_indices:\n", state_indices)
    
    log_probstates = log_prob(params, state_indices)
    print("log_probstates:\n", log_probstates)
    
    state_indices = jnp.ones((sequence_length, ), dtype=jnp.int64)
    logits = _logits(params, state_indices)
    log_softmax_logits = jax.nn.log_softmax(logits, axis=-1)
    print("logits.shape: ", logits.shape)
    print("logits:\n", logits)
    print("logits in softmax:\n", log_softmax_logits)
    print("logits sum:\n", jnp.sum(jnp.exp(log_softmax_logits), axis=-1))
    print(logits.shape)
    
## python3 /mnt/t02home/MLCodes/lithium/src/sampler.py
