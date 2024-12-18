import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
import time

####################################################################################################
# MCMC: Markov Chain Monte Carlo sampling algorithm
# This function samples the variable `x` and generates corresponding state indices.
from .mcmc import mcmc_mode

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, 0,
                            None, 0, 0, None,
                            None, None, None),
                   static_broadcasted_argnums=(2))
def sample_x_mcmc(keys, state_indices,
                  logp, x, params_flw, w_indices,
                  mc_steps, mc_stddev, index_list):
    """
        This function samples the variable `x` and generates corresponding state indices.
        Outputs:
            - x : ndarray, shape=(batch_per_device, num_modes, dim)
                    The newly sampled values of `x`.
    """
    batch_per_device, num_modes, _ = x.shape
    invsqrtw  = 1/jnp.sqrt(w_indices)
    keys, key_state, key_mcmc = jax.random.split(keys, 3)
    
    state_indices_expanded = index_list[state_indices].reshape(batch_per_device, num_modes)
    logp_fn = lambda x: logp(x, params_flw, state_indices_expanded, w_indices)
    x, accept_rate = mcmc_mode(logp_fn, x, key_mcmc, mc_steps, mc_stddev, invsqrtw)
    return keys, x, accept_rate

####################################################################################################

## for nvt & npt ensemble
def make_loss(effective_mass, beta, index_list,
              log_prob, logpsi, logpsi_grad_laplacian, 
              potential_energy, trans_Q2R):

    def observablefn(params_van, params_flw,
                     state_indices, x, w_indices, box_lengths, 
                     R0, Pmat, keys):
        
        #========== calculate E_local & E_mean ==========
        batch_per_device, num_modes, _ = x.shape
        
        logp_states = log_prob(params_van, state_indices)
        state_indices_expanded = index_list[state_indices].reshape(batch_per_device, num_modes)
        
        grad, laplacian = logpsi_grad_laplacian(x, params_flw, state_indices_expanded, w_indices, keys)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)
        
        ### transform to real space
        coord = trans_Q2R(x, R0, box_lengths, Pmat)
        print("coord.shape:", coord.shape)
        
        kinetic = (- (0.5 / effective_mass) * (laplacian + (grad**2).sum(axis=(-2, -1)))).real
        potentl = (potential_energy(coord, box_lengths)).real
        Eloc = jax.lax.stop_gradient(kinetic + potentl)
        print("K.shape:", kinetic.shape)
        print("V.shape:", potentl.shape)
        print("Eloc.shape:", Eloc.shape)
        print("logp_states.shape:", logp_states.shape)

        K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, logprob_states = \
                    jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                           (kinetic.mean(),      (kinetic**2).mean(),
                            potentl.mean(),      (potentl**2).mean(),
                            Eloc.mean(),         (Eloc**2).mean(),
                            logp_states.mean(),
                            ))

        observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                      "V_mean": V_mean, "V2_mean": V2_mean,
                      "E_mean": E_mean, "E2_mean": E2_mean,
                      "logprob_states": logprob_states,
                      }

        return observable

    return observablefn



####################################################################################################
## update class model and quant model (params_van and params_flw)
@partial(jax.pmap, axis_name="p",
        in_axes =(0, 0,
                  0, 0, None, None, None, None, 0,
                  0, None, None, None,
                  ),
        out_axes=(0
                  ),
        static_broadcasted_argnums=(10, 11, 12))

def update_observable(params_van, params_flw, 
                      state_indices, x, w_indices, box_lengths, R0, Pmat, keys, 
                      datas_acc, acc_steps, final_step, observablefn):
    
    #========== calculate gradient ==========
    datas = observablefn(params_van, params_flw, 
                        state_indices, x, w_indices, box_lengths, R0, Pmat, keys)
    datas_acc = jax.tree.map(lambda acc, i: acc + i, datas_acc, datas)

    #========== update at final step ==========
    if final_step:
        datas_acc =jax.tree.map(lambda acc: acc / acc_steps, datas_acc)

    return datas_acc

####################################################################################################

## calculate means and stds for quantities
def calculate_means_and_stds(data, num_atoms, batch, acc_steps):
    # Define constants
    Kelvin_2_meV = 0.08617333262145   # Conversion factor from Kelvin to meV
    # Kelvin_2_GPa = 0.01380649         # Conversion factor from K/A^3 to GPa
    # num_atoms_inv = 1 / num_atoms     # Inverse of the number of atoms for normalization
    batch_acc_inv = 1 / (batch * acc_steps)  # Inverse of batch times accumulation steps

    # List of variables to process
    thermal_vars = ["E", "K", "V"]  # Thermodynamic quantities

    # Calculate mean, standard deviation, and apply unit conversion
    computed_quantities = {}
    
    for var in thermal_vars:
        mean, mean2 = data[f"{var}_mean"], data[f"{var}2_mean"]
        std = jnp.sqrt((mean2 - mean**2) * batch_acc_inv)
        mean, std = mean * Kelvin_2_meV, std * Kelvin_2_meV
        computed_quantities[var] = (mean, std)

    return computed_quantities

####################################################################################################
    
def get_index_basefre(sequence_length, indices_group): 
    """
        Get indices of base frequency, i.e.:
            [0 0 0 0 0 0 0 0 0 0 0 0]
            [1 0 0 0 0 0 0 0 0 0 0 0]
            [0 1 0 0 0 0 0 0 0 0 0 0]
            [0 0 1 0 0 0 0 0 0 0 0 0]
            ...
            [0 0 0 0 0 0 0 0 0 0 0 1]
    """

    num_modes = sequence_length * indices_group
    index_basefre = np.zeros((num_modes+1, sequence_length), dtype=np.int64)
    
    for ix in range(num_modes):
        iy  = ix // indices_group
        idx = indices_group - ix % indices_group
        index_basefre[ix+1, iy] = idx
        
    return jnp.array(index_basefre, dtype=jnp.int64)

####################################################################################################




