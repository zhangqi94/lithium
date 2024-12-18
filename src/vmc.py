import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from functools import partial
import time

####################################################################################################
# MCMC: Markov Chain Monte Carlo sampling algorithm
# This function samples the variable `x` and generates corresponding state indices.
# from .mcmc import mcmcw
from .mcmc import mcmc_mode

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, 
                            None, 0,
                            None, 0, 0, None,
                            None, None, None),
                   static_broadcasted_argnums=(1, 3))
def sample_stateindices_and_x(keys, 
                  sampler, params_van,
                  logp, x, params_flw, w_indices,
                  mc_steps, mc_stddev, index_list):
    """
        This function samples the variable `x` and generates corresponding state indices.
        Outputs:
            - x : ndarray, shape=(batch_per_device, num_modes, dim)
                    The newly sampled values of `x`.
            - state_indices : ndarray, shape=(batch_per_device, num_modes//indices_group)
                    The sampled state indices for the given batch.
    """
    batch_per_device, num_modes, _ = x.shape
    invsqrtw  = 1 / jnp.sqrt(w_indices)
    keys, key_state, key_mcmc = jax.random.split(keys, 3)
    
    state_indices = sampler(params_van, key_state, batch_per_device)
    state_indices_expanded = index_list[state_indices].reshape(batch_per_device, num_modes)
    logp_fn = lambda x: logp(x, params_flw, state_indices_expanded, w_indices)
    x, accept_rate = mcmc_mode(logp_fn, x, key_mcmc, mc_steps, mc_stddev, invsqrtw)
    return keys, state_indices, x, accept_rate


####################################################################################################
## for nvt & npt ensemble
def make_loss(effective_mass, beta, index_list, clip_factor,
              cal_stress, pressurefn, stressfn, dp_energyfn, 
              log_prob, logpsi, logpsi_grad_laplacian, 
              potential_energy, trans_Q2R):

    def observable_and_lossfn(params_van, params_flw, 
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
        
        kinetic = (- (0.5 /effective_mass) * (laplacian + (grad**2).sum(axis=(-2, -1)))).real
        potentl = (potential_energy(coord, box_lengths)).real
        Eloc = jax.lax.stop_gradient(kinetic + potentl)
        Floc = jax.lax.stop_gradient(logp_states / beta + Eloc)
        print("K.shape:", kinetic.shape)
        print("V.shape:", potentl.shape)
        print("Eloc.shape:", Eloc.shape)
        print("logp_states.shape:", logp_states.shape)
        print("Floc.shape:", Floc.shape)
        
        ### pressure and stress are in the unit of K/A^3
        if cal_stress == 0: # do not calculate stress
            pressure = pressurefn(dp_energyfn, coord, box_lengths, kinetic)
            stress = jnp.repeat(pressure[:, jnp.newaxis], 3, axis=1)
            Gloc = Floc + pressure * jnp.prod(box_lengths)
            print("Gloc.shape:", Gloc.shape)
            print("P.shape:", pressure.shape)
            print("Do not calculate the stress!")
            
        elif cal_stress == 1: # stress forall directions are same
            pressure = pressurefn(dp_energyfn, coord, box_lengths, kinetic)
            stress = jnp.repeat(pressure[:, jnp.newaxis], 3, axis=1)
            Gloc = Floc + pressure * jnp.prod(box_lengths)
            print("Gloc.shape:", Gloc.shape)
            print("P.shape:", pressure.shape)
            print("T.shape:", stress.shape)
            print("Calculate the isotropic stress! (all directions are same)")
        
        elif cal_stress == 2: # calculate stress for each direction
            forloop = True
            stress = stressfn(dp_energyfn, coord, box_lengths, kinetic, forloop)
            pressure = stress.mean(axis=1)
            Gloc = Floc + pressure * jnp.prod(box_lengths)
            print("Gloc.shape:", Gloc.shape)
            print("P.shape:", pressure.shape)
            print("T.shape:", stress.shape)
            print("Calculate the anisotropic stress! (three directions are different)")

        K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, \
        S_mean,  S2_mean,  F_mean,  F2_mean,   \
        G_mean,  G2_mean,  P_mean,  P2_mean,  T_mean,  T2_mean  = \
                                jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                                       (kinetic.mean(),      (kinetic**2).mean(),
                                        potentl.mean(),      (potentl**2).mean(),
                                        Eloc.mean(),         (Eloc**2).mean(),
                                        -logp_states.mean(), (logp_states**2).mean(),
                                        Floc.mean(),         (Floc**2).mean(),
                                        Gloc.mean(),         (Gloc**2).mean(),
                                        pressure.mean(),     (pressure**2).mean(),
                                        stress.mean(axis=0), (stress**2).mean(axis=0),
                                        ))

        observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                      "V_mean": V_mean, "V2_mean": V2_mean,
                      "E_mean": E_mean, "E2_mean": E2_mean,
                      "S_mean": S_mean, "S2_mean": S2_mean,
                      "F_mean": F_mean, "F2_mean": F2_mean,
                      "G_mean": G_mean, "G2_mean": G2_mean,
                      "P_mean": P_mean, "P2_mean": P2_mean,
                      "T_mean": T_mean, "T2_mean": T2_mean,
                      }

        #========== calculate classical gradient ==========
        def class_lossfn(params_van):
            logp_states = log_prob(params_van, state_indices)
            
            tv = jax.lax.pmean(jnp.abs(Floc - F_mean).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F_mean - clip_factor*tv, F_mean + clip_factor*tv)
            gradF_phi = (logp_states * Floc_clipped).mean()
            class_score = logp_states.mean()
            return gradF_phi, class_score
        
        #========== calculate quantum gradient ==========
        def quant_lossfn(params_flw):
            logpsix = logpsi(x, params_flw, state_indices_expanded, w_indices)

            tv = jax.lax.pmean(jnp.abs(Eloc - E_mean).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, E_mean - clip_factor*tv, E_mean + clip_factor*tv)
            gradF_theta = 2 * ((logpsix * Eloc_clipped.conj()).real.mean())
            quant_score = 2 * (logpsix.real.mean())
            return gradF_theta, quant_score

        return observable, class_lossfn, quant_lossfn

    return observable_and_lossfn

####################################################################################################
## update class model and quant model (params_van and params_flw)
@partial(jax.pmap, axis_name="p",
        in_axes =(0, 0, None, 
                  0, 0, None, None, None, None, 0,
                  0, 0, 0, 0, 
                  None, None, None, None,
                  ),
        out_axes=(0, 0, None, 0, 
                  0, 0, 0,
                  ),
        static_broadcasted_argnums=(14, 15, 16, 17))

def update_van_flw(params_van, params_flw, opt_state,
                   state_indices, x, w_indices, box_lengths, R0, Pmat, keys, 
                   datas_acc, grads_acc, class_score_acc, quant_score_acc, 
                   acc_steps, final_step, observable_and_lossfn, optimizer
                   ):
    
    #========== calculate gradient ==========
    datas, class_lossfn, quant_lossfn = observable_and_lossfn(params_van, params_flw, 
                                                              state_indices, x, w_indices, box_lengths, 
                                                              R0, Pmat, keys)

    grad_params_van, class_score = jax.jacrev(class_lossfn)(params_van)
    grad_params_flw, quant_score = jax.jacrev(quant_lossfn)(params_flw)
    
    grads = {'van': grad_params_van, 'flw': grad_params_flw}
    
    grads, class_score, quant_score = jax.lax.pmean((grads, class_score, quant_score), 
                                                    axis_name="p"
                                                    )
                    
    datas_acc, grads_acc, class_score_acc, quant_score_acc = jax.tree.map(lambda acc, i: acc + i, 
                                            (datas_acc, grads_acc, class_score_acc, quant_score_acc), 
                                            (datas,     grads,     class_score,     quant_score)
                                            )

    #========== update at final step ==========
    if final_step:
        datas_acc, grads_acc, class_score_acc, quant_score_acc = jax.tree.map(lambda acc: acc / acc_steps, 
                                            (datas_acc, grads_acc, class_score_acc, quant_score_acc)
                                            )
                        
        grad_params_van, grad_params_flw = grads_acc['van'], grads_acc['flw']
        
        grad_params_van = jax.tree.map(lambda grad, class_score: 
                                    grad - datas_acc["F_mean"] * class_score,
                                    grad_params_van, class_score_acc
                                    )
        grad_params_flw = jax.tree.map(lambda grad, quant_score: 
                                    grad - datas_acc["E_mean"] * quant_score,
                                    grad_params_flw, quant_score_acc
                                    )

        grads_acc = {'van': grad_params_van, 'flw': grad_params_flw}
        updates, opt_state = optimizer.update(grads_acc, opt_state)
        params = {'van': params_van, 'flw': params_flw}
        params = optax.apply_updates(params, updates)
        params_van, params_flw = params['van'], params['flw']

    return params_van, params_flw, opt_state, datas_acc, \
           grads_acc, class_score_acc, quant_score_acc

####################################################################################################
## update cell
def update_cell(box_lengths, R0, relax_lr, T, target_pressure, 
                lattice_type, effective_mass, hessian_type, coordinate_type,
                shift_vectors, supercell_size, atom_per_unitcell, 
                get_gradient_and_hessian, init_coordinate_transformations, 
                dp_energyfn
                ):

    # Store old box lengths and compute the updated box lengths
    box_lengths_old = box_lengths.copy()
    box_lengths = box_lengths_old + relax_lr * (T - target_pressure)

    # Adjust box lengths for specific lattice types
    if lattice_type in {"cI16", "fcc", "bcc"}:
        average_value = box_lengths.mean()
        box_lengths = jnp.full_like(box_lengths, average_value)
        
    elif lattice_type in {"tI20"}:
        average_of_first_two = box_lengths[:2].mean()
        box_lengths = box_lengths.at[:2].set(average_of_first_two) 
    
    elif lattice_type in {"oC88", "oC40", "oP48", "oP192"}:
        box_lengths = box_lengths
    
    else:
        raise ValueError("Unsupported lattice type: {}".format(lattice_type))

    # Rescale R0 based on the updated box lengths
    R0 = R0 / box_lengths_old * box_lengths
    num_atoms, dim = R0.shape
    
    # Recompute gradient, Hessian
    V0, Vgrad, Dmat = get_gradient_and_hessian(R0, 
                                               box_lengths, 
                                               effective_mass, 
                                               dp_energyfn, 
                                               hessian_type=hessian_type
                                               )

    # Recompute modes
    wsquare_indices, w_indices, Pmat, wkd_rawdata, num_modes \
                = init_coordinate_transformations(coordinate_type, 
                                                  Dmat, 
                                                  shift_vectors, 
                                                  supercell_size, 
                                                  atom_per_unitcell, 
                                                  num_atoms, 
                                                  dim, 
                                                  cal_kpoint = False, 
                                                  verbosity = 0
                                                  )

    return box_lengths_old, box_lengths, R0, wsquare_indices, w_indices, Pmat

####################################################################################################
## calculate means and stds for quantities
def calculate_means_and_stds(data, num_atoms, batch, acc_steps):
    # Define constants
    Kelvin_2_meV = 0.08617333262145   # Conversion factor from Kelvin to meV
    Kelvin_2_GPa = 0.01380649         # Conversion factor from K/A^3 to GPa
    num_atoms_inv = 1 / num_atoms     # Inverse of the number of atoms for normalization
    batch_acc_inv = 1 / (batch * acc_steps)  # Inverse of batch times accumulation steps

    # List of variables to process
    thermal_vars = ["F", "E", "K", "V", "G"]  # Thermodynamic quantities
    entropy_vars = ["S"]                      # Entropy-related quantities
    pressure_vars = ["P", "T"]                # Pressure-related quantities

    # Calculate mean, standard deviation, and apply unit conversion
    computed_quantities = {}
    
    for var in thermal_vars:
        mean, mean2 = data[f"{var}_mean"], data[f"{var}2_mean"]
        std = jnp.sqrt((mean2 - mean**2) * batch_acc_inv)
        mean, std = mean * num_atoms_inv * Kelvin_2_meV, std * num_atoms_inv * Kelvin_2_meV
        computed_quantities[var] = (mean, std)
    
    for var in entropy_vars:
        mean, mean2 = data[f"{var}_mean"], data[f"{var}2_mean"]
        std = jnp.sqrt((mean2 - mean**2) * batch_acc_inv)
        mean, std = mean * num_atoms_inv, std * num_atoms_inv
        computed_quantities[var] = (mean, std)

    for var in pressure_vars:
        mean, mean2 = data[f"{var}_mean"], data[f"{var}2_mean"]
        std = jnp.sqrt((mean2 - mean**2) * batch_acc_inv)
        mean, std = mean * Kelvin_2_GPa, std * Kelvin_2_GPa
        computed_quantities[var] = (mean, std)

    return computed_quantities

####################################################################################################
## stores the current stress and pressure values.
def store_recent_stress(recent_T_vals, recent_P_vals, T_vals, P_vals, num_recent_vals):
    
    def update_recent_vals(recent_vals, new_vals, num_recent_vals):
        if recent_vals is None: 
            recent_vals = jnp.array([new_vals])
        else:
            recent_vals = jnp.concatenate([recent_vals, jnp.array([new_vals])])
        # Keep only the most recent 'num_recent_vals' entries
        if recent_vals.shape[0] > num_recent_vals:
            recent_vals = recent_vals[-num_recent_vals:]
        return recent_vals
    
    # Extract stress and pressure from T_vals, P_vals.
    T, T_std = T_vals
    P, P_std = P_vals
    # Update recent_T_vals and recent_P_vals
    recent_T_vals = update_recent_vals(recent_T_vals, (T, T_std), num_recent_vals)
    recent_P_vals = update_recent_vals(recent_P_vals, (P, P_std), num_recent_vals)
    
    return recent_T_vals, recent_P_vals

####################################################################################################
## calculates the mean and standard deviation of the recent stress and pressure values.
def calculate_recent_stress(recent_T_vals, recent_P_vals, num_recent_vals):
    
    # Extract stress and pressure from recent_T_vals, recent_P_vals.
    recent_T_jax, recent_T_std_jax = recent_T_vals[:, 0], recent_T_vals[:, 1]
    recent_P_jax, recent_P_std_jax = recent_P_vals[:, 0], recent_P_vals[:, 1]
    
    # Calculate mean and standard deviation of recent_T and recent_P.
    recent_T     = jnp.mean(recent_T_jax, axis=0)
    recent_T_std = jnp.mean(recent_T_std_jax, axis=0) / jnp.sqrt(num_recent_vals)
    recent_P     = jnp.mean(recent_P_jax)
    recent_P_std = jnp.mean(recent_P_std_jax) / jnp.sqrt(num_recent_vals)

    return recent_T, recent_T_std, recent_P, recent_P_std

####################################################################################################
### This is the end of this file





























####################################################################################################
## nvt backups
# def make_loss_nvt_bak(log_prob, logpsi, logpsi_grad_laplacian, 
#                 potential_energy, clip_factor, effective_mass, alpha, 
#                 beta, index_list, trans_Q2R, stressfn, dp_energyfn, box_lengths):

#     def observable_and_lossfn(params_van, params_flw, 
#                               state_indices, x, keys,
#                               ):
#         #========== calculate E_local & E_mean ==========
#         batch_per_device, num_modes, _ = x.shape
        
#         logp_states = log_prob(params_van, state_indices)
#         state_indices_expanded = index_list[state_indices].reshape(batch_per_device, num_modes)
        
#         grad, laplacian = logpsi_grad_laplacian(x, params_flw, state_indices_expanded, keys)
#         print("grad.shape:", grad.shape)
#         print("laplacian.shape:", laplacian.shape)
        
#         kinetic = (- (0.5 * alpha/effective_mass) \
#             * (laplacian + (grad**2).sum(axis=(-2, -1)))).real
#         potentl = ((1/effective_mass) * potential_energy(x)).real
#         Eloc = jax.lax.stop_gradient(kinetic + potentl)
#         Floc = jax.lax.stop_gradient(logp_states / beta + Eloc)
#         print("K.shape:", kinetic.shape)
#         print("V.shape:", potentl.shape)
#         print("Eloc.shape:", Eloc.shape)
#         print("logp_states.shape:", logp_states.shape)
#         print("Floc.shape:", Floc.shape)
        
#         ## pressure in the unit of K/A^3
#         coord = trans_Q2R(x)
#         forloop = True
#         stress = stressfn(dp_energyfn, coord, box_lengths, kinetic, forloop)
#         pressure = stress.mean(axis=1)
#         ## pressure = pressurefn(dp_energyfn, coord, box_lengths, kinetic)
        
#         Gloc = Floc + pressure * jnp.prod(box_lengths)
#         print("Gloc.shape:", Gloc.shape)
#         print("P.shape:", pressure.shape)
#         print("T.shape:", stress.shape)

#         K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, \
#         S_mean,  S2_mean,  F_mean,  F2_mean,   \
#         G_mean,  G2_mean,  P_mean,  P2_mean,  T_mean,  T2_mean  = \
#                     jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
#                            (kinetic.mean(),      (kinetic**2).mean(),
#                             potentl.mean(),      (potentl**2).mean(),
#                             Eloc.mean(),         (Eloc**2).mean(),
#                             -logp_states.mean(), (logp_states**2).mean(),
#                             Floc.mean(),         (Floc**2).mean(),
#                             Gloc.mean(),         (Gloc**2).mean(),
#                             pressure.mean(),     (pressure**2).mean(),
#                             stress.mean(axis=0), (stress**2).mean(axis=0),
#                             ))
   
#         observable = {"K_mean": K_mean, "K2_mean": K2_mean,
#                       "V_mean": V_mean, "V2_mean": V2_mean,
#                       "E_mean": E_mean, "E2_mean": E2_mean,
#                       "S_mean": S_mean, "S2_mean": S2_mean,
#                       "F_mean": F_mean, "F2_mean": F2_mean,
#                       "G_mean": G_mean, "G2_mean": G2_mean,
#                       "P_mean": P_mean, "P2_mean": P2_mean,
#                       "T_mean": T_mean, "T2_mean": T2_mean,
#                       }
        
#         def class_lossfn(params_van):
#             logp_states = log_prob(params_van, state_indices)
            
#             tv = jax.lax.pmean(jnp.abs(Floc - F_mean).mean(), axis_name="p")
#             Floc_clipped = jnp.clip(Floc, F_mean - clip_factor*tv, F_mean + clip_factor*tv)
#             gradF_phi = (logp_states * Floc_clipped).mean()
#             class_score = logp_states.mean()
#             return gradF_phi, class_score
        
#         def quant_lossfn(params_flw):
#             logpsix = logpsi(x, params_flw, state_indices_expanded)

#             tv = jax.lax.pmean(jnp.abs(Eloc - E_mean).mean(), axis_name="p")
#             Eloc_clipped = jnp.clip(Eloc, E_mean - clip_factor*tv, E_mean + clip_factor*tv)
#             gradF_theta = 2 * ((logpsix * Eloc_clipped.conj()).real.mean())
#             quant_score = 2 * (logpsix.real.mean())
#             return gradF_theta, quant_score

#         return observable, class_lossfn, quant_lossfn

#     return observable_and_lossfn


    # data = jax.tree.map(lambda x: x[0], datas_acc)
    # accept_rate = accept_rate_acc[0] / acc_steps
    # F, F2_mean = data["F_mean"], data["F2_mean"]
    # E, E2_mean = data["E_mean"], data["E2_mean"]
    # K, K2_mean = data["K_mean"], data["K2_mean"]
    # V, V2_mean = data["V_mean"], data["V2_mean"]
    # S, S2_mean = data["S_mean"], data["S2_mean"]
    # G, G2_mean = data["G_mean"], data["G2_mean"]
    # P, P2_mean = data["P_mean"], data["P2_mean"]
    # T, T2_mean = data["T_mean"], data["T2_mean"]
    # F_std = jnp.sqrt((F2_mean - F**2) / (batch*acc_steps))
    # E_std = jnp.sqrt((E2_mean - E**2) / (batch*acc_steps))
    # K_std = jnp.sqrt((K2_mean - K**2) / (batch*acc_steps))
    # V_std = jnp.sqrt((V2_mean - V**2) / (batch*acc_steps))
    # S_std = jnp.sqrt((S2_mean - S**2) / (batch*acc_steps))
    # G_std = jnp.sqrt((G2_mean - G**2) / (batch*acc_steps))
    # P_std = jnp.sqrt((P2_mean - P**2) / (batch*acc_steps))
    # T_std = jnp.sqrt((T2_mean - T**2) / (batch*acc_steps))
    # # change the unit into meV
    # Kelvin_2_meV = 8.617333262145e-2
    # F, F_std = F/num_atoms * Kelvin_2_meV, F_std/num_atoms * Kelvin_2_meV
    # E, E_std = E/num_atoms * Kelvin_2_meV, E_std/num_atoms * Kelvin_2_meV
    # K, K_std = K/num_atoms * Kelvin_2_meV, K_std/num_atoms * Kelvin_2_meV
    # V, V_std = V/num_atoms * Kelvin_2_meV, V_std/num_atoms * Kelvin_2_meV
    # S, S_std = S/num_atoms,                S_std/num_atoms
    # G, G_std = G/num_atoms * Kelvin_2_meV, G_std/num_atoms * Kelvin_2_meV
    # # change the pressure and stress from K/A^3 to GPa
    # Kelvin_2_GPa = 0.01380649
    # P, P_std = P * Kelvin_2_GPa, P_std * Kelvin_2_GPa
    # T, T_std = T * Kelvin_2_GPa, T_std * Kelvin_2_GPa
    
    


# def make_loss(effective_mass, beta, index_list, alpha, clip_factor,
#               cal_stress, pressurefn, stressfn, dp_energyfn, 
#               log_prob, logpsi, logpsi_grad_laplacian, 
#               potential_energy, trans_Q2R):

#     def observable_and_lossfn(params_van, params_flw, 
#                               state_indices, x, w_indices, box_lengths, 
#                               R0, Pmat, keys):
#         #========== calculate E_local & E_mean ==========
#         batch_per_device, num_modes, _ = x.shape
        
#         logp_states = log_prob(params_van, state_indices)
#         state_indices_expanded = index_list[state_indices].reshape(batch_per_device, num_modes)
        
#         grad, laplacian = logpsi_grad_laplacian(x, params_flw, state_indices_expanded, w_indices, keys)
#         print("grad.shape:", grad.shape)
#         print("laplacian.shape:", laplacian.shape)
        
#         ### transform to real space
#         coord = trans_Q2R(x, R0, box_lengths, Pmat)
#         print("coord.shape:", coord.shape)
        
#         kinetic = (- (0.5 * alpha/effective_mass) \
#             * (laplacian + (grad**2).sum(axis=(-2, -1)))).real
#         potentl = (potential_energy(coord, box_lengths)).real
#         Eloc = jax.lax.stop_gradient(kinetic + potentl)
#         Floc = jax.lax.stop_gradient(logp_states / beta + Eloc)
#         print("K.shape:", kinetic.shape)
#         print("V.shape:", potentl.shape)
#         print("Eloc.shape:", Eloc.shape)
#         print("logp_states.shape:", logp_states.shape)
#         print("Floc.shape:", Floc.shape)
        
#         if cal_stress == 0: # do not calculate stress
#             ### pressure and stress are in the unit of K/A^3
#             pressure = pressurefn(dp_energyfn, coord, box_lengths, kinetic)
            
#             Gloc = Floc + pressure * jnp.prod(box_lengths)
#             print("Gloc.shape:", Gloc.shape)
#             print("P.shape:", pressure.shape)
#             print("Do not calculate the stress!")

#             K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, \
#             S_mean,  S2_mean,  F_mean,  F2_mean,   \
#             G_mean,  G2_mean,  P_mean,  P2_mean,  T_mean,  T2_mean  = \
#                         jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
#                                (kinetic.mean(),      (kinetic**2).mean(),
#                                 potentl.mean(),      (potentl**2).mean(),
#                                 Eloc.mean(),         (Eloc**2).mean(),
#                                 -logp_states.mean(), (logp_states**2).mean(),
#                                 Floc.mean(),         (Floc**2).mean(),
#                                 Gloc.mean(),         (Gloc**2).mean(),
#                                 pressure.mean(),     (pressure**2).mean(),
#                                 jnp.zeros(3),        jnp.zeros(3),
#                                 ))
            
#         elif cal_stress == 1: # stress forall directions are same
#             ### pressure and stress are in the unit of K/A^3
#             pressure = pressurefn(dp_energyfn, coord, box_lengths, kinetic)
#             stress = jnp.repeat(pressure[:, jnp.newaxis], 3, axis=1)
            
#             Gloc = Floc + pressure * jnp.prod(box_lengths)
#             print("Gloc.shape:", Gloc.shape)
#             print("P.shape:", pressure.shape)
#             print("T.shape:", stress.shape)
#             print("Calculate the isotropic stress!")

#             K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, \
#             S_mean,  S2_mean,  F_mean,  F2_mean,   \
#             G_mean,  G2_mean,  P_mean,  P2_mean,  T_mean,  T2_mean  = \
#                         jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
#                                (kinetic.mean(),      (kinetic**2).mean(),
#                                 potentl.mean(),      (potentl**2).mean(),
#                                 Eloc.mean(),         (Eloc**2).mean(),
#                                 -logp_states.mean(), (logp_states**2).mean(),
#                                 Floc.mean(),         (Floc**2).mean(),
#                                 Gloc.mean(),         (Gloc**2).mean(),
#                                 pressure.mean(),     (pressure**2).mean(),
#                                 stress.mean(axis=0), (stress**2).mean(axis=0),
#                                 ))
        
#         elif cal_stress == 2: # calculate stress for each direction
#             ### pressure and stress are in the unit of K/A^3
#             forloop = True
#             stress = stressfn(dp_energyfn, coord, box_lengths, kinetic, forloop)
#             pressure = stress.mean(axis=1)

#             Gloc = Floc + pressure * jnp.prod(box_lengths)
#             print("Gloc.shape:", Gloc.shape)
#             print("P.shape:", pressure.shape)
#             print("T.shape:", stress.shape)
#             print("Calculate the anisotropic stress!")

#             K_mean,  K2_mean,  V_mean,  V2_mean,  E_mean,  E2_mean, \
#             S_mean,  S2_mean,  F_mean,  F2_mean,   \
#             G_mean,  G2_mean,  P_mean,  P2_mean,  T_mean,  T2_mean  = \
#                         jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
#                                (kinetic.mean(),      (kinetic**2).mean(),
#                                 potentl.mean(),      (potentl**2).mean(),
#                                 Eloc.mean(),         (Eloc**2).mean(),
#                                 -logp_states.mean(), (logp_states**2).mean(),
#                                 Floc.mean(),         (Floc**2).mean(),
#                                 Gloc.mean(),         (Gloc**2).mean(),
#                                 pressure.mean(),     (pressure**2).mean(),
#                                 stress.mean(axis=0), (stress**2).mean(axis=0),
#                                 ))

#         observable = {"K_mean": K_mean, "K2_mean": K2_mean,
#                     "V_mean": V_mean, "V2_mean": V2_mean,
#                     "E_mean": E_mean, "E2_mean": E2_mean,
#                     "S_mean": S_mean, "S2_mean": S2_mean,
#                     "F_mean": F_mean, "F2_mean": F2_mean,
#                     "G_mean": G_mean, "G2_mean": G2_mean,
#                     "P_mean": P_mean, "P2_mean": P2_mean,
#                     "T_mean": T_mean, "T2_mean": T2_mean,
#                     }

#         #========== calculate classical gradient ==========
#         def class_lossfn(params_van):
#             logp_states = log_prob(params_van, state_indices)
            
#             tv = jax.lax.pmean(jnp.abs(Floc - F_mean).mean(), axis_name="p")
#             Floc_clipped = jnp.clip(Floc, F_mean - clip_factor*tv, F_mean + clip_factor*tv)
#             gradF_phi = (logp_states * Floc_clipped).mean()
#             class_score = logp_states.mean()
#             return gradF_phi, class_score
        
#         #========== calculate quantum gradient ==========
#         def quant_lossfn(params_flw):
#             logpsix = logpsi(x, params_flw, state_indices_expanded, w_indices)

#             tv = jax.lax.pmean(jnp.abs(Eloc - E_mean).mean(), axis_name="p")
#             Eloc_clipped = jnp.clip(Eloc, E_mean - clip_factor*tv, E_mean + clip_factor*tv)
#             gradF_theta = 2 * ((logpsix * Eloc_clipped.conj()).real.mean())
#             quant_score = 2 * (logpsix.real.mean())
#             return gradF_theta, quant_score

#         return observable, class_lossfn, quant_lossfn

#     return observable_and_lossfn