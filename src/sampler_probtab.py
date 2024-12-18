import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import haiku as hk
import itertools

####################################################################################################
def make_probtab_sampler(van, 
                         num_levels, 
                         sequence_length, 
                         indices_group, 
                         w_indices_init, 
                         beta,
                         output_logits = False,
                         ):
    """
    Covert from make_autoregressive_sampler.py!
    
    Constructs an sampler from a probability table.

    Inputs:
        - van (object): The variational autoregressive network (VAN) to generate logits.
        - num_levels (int): The number of discrete levels for each element in the sequence.
        - sequence_length (int): The length of the sequence to be generated.
        - indices_group (int): The size of the index group for generating combinations.
        - w_indices_init (array): Frequency indices for initializing the sampler.
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
    
    group_w = w_indices_init.reshape(sequence_length, indices_group)
    beta_bands = beta * jnp.einsum("id,jd->ij", group_w, index_list)
    # print("beta_bands:\n", beta_bands)
    
    #========== get logits from van ==========
    def _logits(params):
        #### logits.shape: (sequence_length, num_levels**indices_group)
        logits = van.apply(params)
        logits = logits - beta_bands
        return logits

    #========== make sampler (foriloop version) ==========
    def sampler(params, key, batch):
        key, subkey = jax.random.split(key)
        logits = _logits(params)
        logits = jnp.broadcast_to(logits, (batch,) + logits.shape)

        state_indices = jax.random.categorical(subkey, logits, axis=-1)
        return state_indices

    #========== calculate log_prob for states ==========
    def log_prob(params, state_indices):
        logits = _logits(params)
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

    from probtab import make_probability_table
    from jax.flatten_util import ravel_pytree
    
    key = jax.random.key(42) 
    num_levels = 2
    indices_group = 3
    sequence_length = 16
    beta = 0.1
    batch = 6
    
    van = make_probability_table(num_levels, 
                                 indices_group, 
                                 sequence_length
                                 )
    
    params = van.init(key)
    logits = van.apply(params)
    raveled_params, _ = ravel_pytree(params)
    print("#parameters in the model: %d" % raveled_params.size, flush=True)
    print("logits.shape", logits.shape)
    print("logits:\n", logits)

    num_modes = sequence_length * indices_group
    w_indices_init = jax.random.uniform(key, (num_modes,), minval=1, maxval=2)
    # print("w:", w_indices_init)
    
    sampler, log_prob_novmap, index_list, _logits = make_probtab_sampler(van, 
                                                                         num_levels, 
                                                                         sequence_length, 
                                                                         indices_group, 
                                                                         w_indices_init, 
                                                                         beta, 
                                                                         output_logits=True
                                                                         )
    
    log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)
    state_indices = sampler(params, key, batch)
    print("state_indices:\n", state_indices)
    
    log_probstates = log_prob(params, state_indices)
    print("log_probstates:\n", log_probstates)
    
    logits = _logits(params)
    log_softmax_logits = jax.nn.log_softmax(logits, axis=-1)
    print("logits:\n", logits)
    print("logits in softmax:\n", log_softmax_logits)
    print("logits sum:\n", jnp.sum(jnp.exp(log_softmax_logits), axis=-1))
    print(logits.shape)

    # logits = jnp.broadcast_to(logits, (batch,) + logits.shape)
    # print(jnp.allclose(logits[0], logits[1]))


## python3 /mnt/t02home/MLCodes/lithium/src/sampler_probtab.py
