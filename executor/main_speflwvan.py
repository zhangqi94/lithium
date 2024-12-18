import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import haiku
import flax
import os
import sys
import time


####################################################################################################
def main_speflwvan(args):

    print("Measuring the phonon energy spectrum with van and flow model.")
        
    #========== params args ==========
    load_ckpt, hessian_type = args.load_ckpt, args.hessian_type
    mc_therm, mc_steps, mc_stddev = args.mc_therm, args.mc_steps, args.mc_stddev
    batch, acc_steps, seed = args.batch, args.acc_steps, args.seed
    hutchinson = args.hutchinson
    epoch = args.epoch
    num_devices = args.num_devices

####################################################################################################
    print("\n========== Initialize check point ==========")
    from src.checkpoint import ckpt_filename, save_pkl_data, load_pkl_data
    print("load pkl data from:\n", load_ckpt)
    ckpt = load_pkl_data(load_ckpt)
    R0 = ckpt["R0"]
    box_lengths = ckpt["box_lengths"]
    Pmat = ckpt["Pmat"]
    w_indices_load = ckpt["w_indices"]
    w_indices_init = ckpt["w_indices_init"]
    num_atoms = ckpt["num_atoms"]
    num_modes = ckpt["num_modes"]
    pkl_args = ckpt["args"]

    #========== open file ==========
    file_name = load_ckpt[:-4] + "specflwvan.txt"
    print("#save file name:\n", file_name, flush=True)
    f = open(file_name, "w", buffering=1, newline="\n")

    #========== params pkl args ==========
    isotope, dpfile, lattice_type = pkl_args.isotope, pkl_args.dpfile, pkl_args.lattice_type
    ensemble = pkl_args.ensemble
    coordinate_type = pkl_args.coordinate_type
    dim = pkl_args.dim
    temperature, num_levels, indices_group = pkl_args.temperature, pkl_args.num_levels, pkl_args.indices_group
    van_type = pkl_args.van_type
    van_layers, van_size = pkl_args.van_layers, pkl_args.van_size 
    van_heads, van_hidden = pkl_args.van_heads, pkl_args.van_hidden
    flow_type = pkl_args.flow_type
    flow_depth, mlp_depth, mlp_width = pkl_args.flow_depth, pkl_args.mlp_depth, pkl_args.mlp_width
    flow_st = pkl_args.flow_st
    hessian_type = pkl_args.hessian_type
    hutchinson = pkl_args.hutchinson

    key = jax.random.key(args.seed)
    
    #========== useful constants & quantities ==========
    Kelvin_2_meV = 0.08617333262145   # Conversion factor from Kelvin to meV
    # Kelvin_2_GPa = 0.01380649         # Conversion factor from K/A^3 to GPa
    
    beta = 1/temperature
    
    #========== gpu ==========
    print("jax.random.key:", args.seed)
    print("jax.devices:", jax.devices(), flush=True)
    if num_devices == jax.device_count():
        print("number of GPU devices:", num_devices)
    else:
        raise ValueError("number of GPU devices is not equal to num_devices")

    if batch % num_devices != 0:
        raise ValueError("Batch size must be divisible by the number of GPU devices. "
                            "Got batch %d for %d devices now." % (batch, num_devices))
    else:
        batch_per_device = batch // num_devices
        print("Total batch = %d for %d devices, batch_per_device = %d" 
                            % (batch, num_devices, batch_per_device))
        
####################################################################################################
    print("\n========== Initialize lattice ==========")
    from src.crystal_lithium import get_phys_const, create_supercell, estimated_volume_from_pressure
    h2_over_2m, effective_mass = get_phys_const(isotope)
    
    volume_per_atom = pkl_args.volume_per_atom
    supercell_size = pkl_args.supercell_size
    supercell_length = pkl_args.supercell_length
    _, _, _, shift_vectors, atom_per_unitcell = create_supercell(lattice_type, 
                                                                volume_per_atom, 
                                                                supercell_size, 
                                                                supercell_length
                                                                )

    print("isotope:", isotope)
    print("effective mass: %.6f" % effective_mass)
    print("hbar^2/(2m): %.6f (K)" % h2_over_2m)
    print("lattice type:", lattice_type)
    print("number of atoms: %d" % num_atoms)
    print("supercell lengths:", box_lengths, "(A)")
    print("supercell volume: %.6f (A^3)" % jnp.prod(box_lengths))
    print("temperature: %.6f (K),  beta: %.6f" % (temperature, beta))

####################################################################################################
    print("\n========== Initialize potential energy surface ==========")
    from src.dpjax_lithium import make_dp_model, pkl_mapping, read_pkl_params
    pkl_name = pkl_mapping.get(dpfile)
    dp_energyfn, _ = make_dp_model(pkl_name, num_atoms, box_lengths_init=box_lengths, unit="K")
    Vtot0 = dp_energyfn(R0, box_lengths)
    potential_energy = jax.vmap(dp_energyfn, in_axes=(0, None), out_axes=(0)) 

    print("initialize potential energy of", isotope)
    print("load deepmd model", dpfile, "file from:", pkl_name)
    read_pkl_params(pkl_name, verbosity=2)
    
    print("at equilibrium position: V0 = %.6f (K/atom), %.6f (meV/atom)" 
        %(Vtot0/num_atoms, Vtot0/num_atoms*Kelvin_2_meV), flush=True)

####################################################################################################
    print("\n========== Calculate gradient & hessian of dynamic matrix ==========")
    from src.coordtrans_phon import get_gradient_and_hessian
    V0, Vgrad, Dmat = get_gradient_and_hessian(R0, 
                                               box_lengths, 
                                               effective_mass, 
                                               dp_energyfn, 
                                               hessian_type=hessian_type)

    print("crystal properties: (in the unit of mass)")
    print("potential energy V0: %.6f,  per atom: %.6f" %(V0, V0/num_atoms))
    print("V_gradient (first 4 terms):", Vgrad[0:4], "...")
    print("    gradient close to zero (abs<1e-6): %d,    total: %d" 
                            %(jnp.sum(jnp.abs(Vgrad)<1e-6),  Vgrad.shape[0]))

    print("V_hessian (dynamic matrix):", Dmat)
    print("    diagional elements:", jnp.diag(Dmat)[0:4], "...", flush=True)

    ####################################################################################################
    print("\n========== Initial coordinate transformations ==========")
    from src.coordtrans_phon import init_coordinate_transformations, get_coordinate_transforms
    
    wsquare_indices, w_indices, Pmat, wkd_rawdata, num_modes \
        = init_coordinate_transformations(coordinate_type, 
                                          Dmat, 
                                          shift_vectors, 
                                          supercell_size, 
                                          atom_per_unitcell, 
                                          num_atoms, 
                                          dim, 
                                          cal_kpoint = False, 
                                          verbosity = 2
                                          )

    print("number of modes:", num_modes)
    print("frequency w: (in the unit of meV)\n", w_indices * (1/effective_mass) * Kelvin_2_meV)

    print("recompute the frequencies of each mode (in meV)")
    for ii in range(num_modes):
        print("idx0: %d    w: %.6f  %.6f" %(ii+1, 
                        w_indices[ii] * (1/effective_mass) * Kelvin_2_meV,
                        w_indices_load[ii] * (1/effective_mass) * Kelvin_2_meV))
        f.write( ("%d    %6f\n") % (ii+1, w_indices[ii] * (1/effective_mass) * Kelvin_2_meV))
        
    f.write(("\n"))

    trans_Q2R_novmap, _ = get_coordinate_transforms(num_atoms, dim, coordinate_type)
    trans_Q2R = jax.vmap(trans_Q2R_novmap, in_axes=(0, None, None, None), out_axes=(0))

####################################################################################################
    # print("\n========== Initialize orbitals ==========")
    from src.orbitals import get_orbitals_1d, get_orbitals_energy
    sp_orbitals, _ = get_orbitals_1d()

####################################################################################################
    print("\n========== Initialize autoregressive model ==========")
    from src.tools import shard, replicate, automatic_mcstddev, convert_params_dtype

    if van_type == "van_tfhaiku":
        #========== make autoregressive model ==========
        from src.autoregressive import make_autoregressive_model
        van = make_autoregressive_model(num_levels, 
                                        indices_group,
                                        van_layers, 
                                        van_size, 
                                        van_heads, 
                                        van_hidden
                                        )
        sequence_length = num_modes//indices_group
        params_van = van.init(key, jnp.zeros((sequence_length, 1), dtype=jnp.float64))
        params_van = convert_params_dtype(params_van, jnp.float64)
        raveled_params_van, _ = ravel_pytree(params_van)
        print("autoregressive model  [num_levels: %d,  sampler group: %d]" 
                                    % (num_levels, indices_group))
        print("                      [layers: %d,  size: %d,  heads: %d,  hidden: %d]" 
                                    %(van_layers, van_size, van_heads, van_hidden))
        print("sequence length: %d,  group levels: %d" 
            %(sequence_length, num_levels**indices_group))
        print("    #parameters in the autoregressive model: %d" % raveled_params_van.size, flush=True)

        #========== make sampler ==========
        from src.sampler import make_autoregressive_sampler
        w_indices_init = ckpt["w_indices_init"]
        sampler, log_prob_novmap, index_list = make_autoregressive_sampler(van, 
                                                                           num_levels, 
                                                                           sequence_length, 
                                                                           indices_group, 
                                                                           w_indices_init, 
                                                                           beta
                                                                           )
        log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)

    elif van_type == "van_probtab":
        #========== make probability table ==========
        from src.probtab import make_probability_table
        sequence_length = num_modes//indices_group
        van = make_probability_table(num_levels, 
                                     indices_group, 
                                     sequence_length
                                     )
        params_van = van.init(key)
        params_van = convert_params_dtype(params_van, jnp.float64)
        raveled_params_van, _ = ravel_pytree(params_van)
        print("probability table:  [num_levels: %d,  sampler group: %d]" )
        print("    #parameters in the probability table: %d" % raveled_params_van.size, flush=True)

        #========== make sampler ==========
        from src.sampler_probtab import make_probtab_sampler
        w_indices_init = ckpt["w_indices_init"]
        sampler, log_prob_novmap, index_list = make_probtab_sampler(van, 
                                                                    num_levels, 
                                                                    sequence_length, 
                                                                    indices_group, 
                                                                    w_indices_init, 
                                                                    beta
                                                                    )
        log_prob = jax.vmap(log_prob_novmap, (None, 0), 0)

####################################################################################################
    print("\n========== Initialize flow model & wavefunction ==========")
    if flow_type == "flw_rnvpflax":
        from src.flow_rnvpflax import make_flow_model
        from src.logpsi_rnvpflax_real import make_logpsi, make_logphi_logjacdet, \
                                             make_logp, make_logpsi_grad_laplacian
        flow = make_flow_model(flow_depth, 
                               mlp_width, 
                               mlp_depth, 
                               num_modes, 
                               flow_st
                               )
        print("rnvp model (flax) [depth: %d,  hidden layers: %d, %d]" 
            % (flow_depth, mlp_width, mlp_depth))
        print(f"                  use flow_st = {flow_st} in the flow model.")
        
    elif flow_type == "flw_identity":
        from src.flow_identity import make_flow_model
        from src.logpsi_rnvpflax_real import make_logpsi, make_logphi_logjacdet, \
                                             make_logp, make_logpsi_grad_laplacian
        flow = make_flow_model(num_modes, 
                               flow_st
                               )
        print("identity flow model: scale and shift on each mode.")
        print(f"                  use flow_st = {flow_st} in the flow model.")

    params_flw = flow.init(key, jnp.zeros((num_modes, 1), dtype=jnp.float64))
    params_flw = convert_params_dtype(params_flw, jnp.float64)
    raveled_params_flw, _ = ravel_pytree(params_flw)
    print("    #parameters in the flow model: %d" % raveled_params_flw.size, flush=True)

    #========== logpsi = logphi + 0.5*logjacdet ==========
    if hutchinson:
        print("use Hutchinson trick to calculate logpsi")
    logpsi_novmap = make_logpsi(flow, sp_orbitals)
    logphi, logjacdet = make_logphi_logjacdet(flow, sp_orbitals)
    logp = make_logp(logpsi_novmap)
    logpsi, logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi_novmap, 
                                                               forloop=True, 
                                                               hutchinson=hutchinson, 
                                                               logphi=logphi, 
                                                               logjacdet=logjacdet
                                                               )

    ####################################################################################################
    #========== load optimizer & params ==========
    # print("load params_flw from ckpt", flush=True)
    params_van, params_flw = ckpt["params_van"], ckpt["params_flw"]
    params_van, params_flw = replicate((params_van, params_flw), num_devices)

    x = 0.1 * jax.random.normal(key, (num_devices, batch_per_device, num_modes, 1), dtype=jnp.float64)
    keys = jax.random.split(key, num_devices)
    x, keys = shard(x), shard(keys)
    print("keys shape:", keys.shape)
    print("x.shape:", x.shape)

    #========== base mode index ==========
    from src.vmc_speflwvan import sample_x_mcmc, make_loss, \
                                  update_observable, calculate_means_and_stds

    #========== observable function ==========
    observablefn = make_loss(effective_mass, beta, index_list,
                             log_prob, logpsi, logpsi_grad_laplacian,
                             potential_energy, trans_Q2R)

    ####################################################################################################
    print("\n========== Measuring ==========")
    t0 = time.time()
    print("start measuring:")
    print("    E, K, V (in the unit of meV/atom)")

    for ii in range(epoch):
        tf1 = time.time()
        
        datas_acc = replicate({"E_mean": 0., "E2_mean": 0., # energy
                               "K_mean": 0., "K2_mean": 0., # kinetic energy
                               "V_mean": 0., "V2_mean": 0., # potential energy
                               "logprob_states": 0., # log probability of states
                               }, 
                              num_devices
                              )
        accept_rate_acc = shard(jnp.zeros(num_devices))
        
        if ii == 0:
            state_index = jnp.zeros((sequence_length, ), dtype=jnp.int64)
        else:
            key_state = jax.random.split(keys[0], 10)[5]
            state_index = sampler(ckpt["params_van"], key_state, 1).reshape((sequence_length, ))
        state_indices = jnp.tile(state_index, (num_devices, batch_per_device, 1))
        state_indices = shard(state_indices)
        
        #========== thermalized ==========
        for jj in range(mc_therm):
            #t1 = time.time()
            keys, x, accept_rate = sample_x_mcmc(keys, state_indices,
                                                logp, x, params_flw, w_indices,
                                                mc_steps, mc_stddev, index_list)
            #t2 = time.time()
            accept_rate = jnp.mean(accept_rate)
            # print("---- thermal step: %d,  ac: %.4f,  dx: %.4f,  dt: %.3f ----" 
            # % (jj+1, accept_rate, mc_stddev, t2-t1), flush=True)
            mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)

        for acc in range(acc_steps):   
            keys, x, accept_rate = sample_x_mcmc(keys, state_indices,
                                                logp, x, params_flw, w_indices,
                                                mc_steps, mc_stddev, index_list)

            accept_rate_acc += accept_rate
            final_step = (acc == (acc_steps-1))
            
            datas_acc = update_observable(params_van, params_flw, 
                    state_indices, x, w_indices, box_lengths, R0, Pmat, keys, 
                    datas_acc, acc_steps, final_step, observablefn)
        
        data = jax.tree.map(lambda x: x[0], datas_acc)
        accept_rate = accept_rate_acc[0] / acc_steps
        
        computed_quantities = calculate_means_and_stds(data, num_atoms, batch, acc_steps)
        E, E_std = computed_quantities["E"] # local energy
        K, K_std = computed_quantities["K"] # kinetic energy
        V, V_std = computed_quantities["V"] # potential energy
        logprob_states = data["logprob_states"]
        tf2 = time.time()
        
        ####========== print ==========
        print("idx: %05d" % ii,
            " E: %.3f (%.3f)  K: %.3f (%.3f)  V: %.3f (%.3f)  logp: %.3f"
            % (E, E_std, 
               K, K_std, 
               V, V_std,
               logprob_states,
               ),
            " ac: %.4f  dx: %.6f  dt: %.3f" 
            % (accept_rate, 
               mc_stddev, 
               tf2-tf1
               ), 
            end=" ")
        print("[", " ".join(str(x) for x in state_index), "]", flush=True)
        
        ####========== save txt data ==========
        f.write( ("%6d" + "  %.6f"*6 + "  %.6f" + "  %.6f"*2 + "\n") 
                % (ii, 
                   E, E_std, 
                   K, K_std, 
                   V, V_std, 
                   logprob_states,
                   accept_rate, 
                   mc_stddev
                   )
                )

        mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)

