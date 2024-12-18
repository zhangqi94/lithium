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
    
def main_train(args):
    
    print("Training neural network for solid lithium.")
    
    #========== params savedata ==========
    folder = args.folder
    load_ckpt = args.load_ckpt
    ckpt_epochs = args.ckpt_epochs
    #========== params physical==========
    isotope = args.isotope
    dpfile = args.dpfile
    ensemble = args.ensemble
    target_pressure = args.target_pressure
    volume_per_atom = args.volume_per_atom
    supercell_length = jnp.array(args.supercell_length, dtype=jnp.float64)
    supercell_size = jnp.array(args.supercell_size, dtype=jnp.int64)
    lattice_type = args.lattice_type
    coordinate_type = args.coordinate_type
    dim = args.dim
    hessian_type = args.hessian_type
    #========== params autoregressive ==========
    temperature = args.temperature
    num_levels = args.num_levels
    indices_group = args.indices_group
    van_type = args.van_type
    van_layers = args.van_layers
    van_size = args.van_size
    van_heads = args.van_heads
    van_hidden = args.van_hidden
    #========== params flow ==========
    flow_type = args.flow_type
    flow_depth = args.flow_depth
    mlp_width = args.mlp_width
    mlp_depth = args.mlp_depth
    flow_st = args.flow_st
    #========== params optimizer ==========
    lr_class = args.lr_class
    lr_quant = args.lr_quant
    min_lr_class = args.min_lr_class
    min_lr_quant = args.min_lr_quant
    decay_rate = args.decay_rate
    decay_steps = args.decay_steps
    decay_begin = args.decay_begin
    hutchinson = args.hutchinson
    clip_factor = args.clip_factor
    cal_stress = args.cal_stress
    #========== params relaxation ==========
    relax_begin = args.relax_begin
    relax_steps = args.relax_steps
    relax_therm = args.relax_therm
    relax_lr = args.relax_lr
    relax_min_lr = args.relax_min_lr
    relax_decay = args.relax_decay
    num_recent_vals = args.num_recent_vals
    #========== params thermal ==========
    mc_therm = args.mc_therm
    mc_steps = args.mc_steps
    mc_stddev = args.mc_stddev
    #========== params training ==========
    batch = args.batch
    acc_steps = args.acc_steps
    epoch = args.epoch
    num_devices = args.num_devices

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
        print("total batch = %d for %d devices, batch_per_device = %d" 
                            % (batch, num_devices, batch_per_device))

    if relax_steps > 0 and cal_stress == 0:
        raise ValueError("relax_steps > 0 requires cal_stress = 1 or 2, now is %d" % cal_stress)
    if relax_steps > 0 and ensemble != "npt":
        raise ValueError("relax_steps > 0 requires ensemble = npt, now is %s" % ensemble)

####################################################################################################
    print("\n========== load check point files ==========")
    from src.checkpoint import ckpt_filename, save_pkl_data, load_pkl_data
    if load_ckpt is not None and load_ckpt != "None":
        print("load pkl data from:\n", load_ckpt)
        ckpt = load_pkl_data(load_ckpt)
    else:
        print("No checkpoint file found. Start from scratch.")

####################################################################################################
    print("\n========== Initialize lattice ==========")
    from src.crystal_lithium import get_phys_const, create_supercell, estimated_volume_from_pressure
    h2_over_2m, effective_mass = get_phys_const(isotope)

    if ensemble == "nvt":
        if volume_per_atom < 0:
            raise ValueError("volume_per_atom must be positive for nvt ensemble!!!")
        print("ensemble: nvt (canonical ensemble)") 
        print("volume per atom: %.6f (A^3)" % volume_per_atom)
        R0, box_lengths, num_atoms, shift_vectors, atom_per_unitcell = create_supercell(lattice_type, 
                                                                                        volume_per_atom, 
                                                                                        supercell_size, 
                                                                                        supercell_length
                                                                                        )

    elif ensemble == "npt":
        if volume_per_atom < 0:
            volume_per_atom = estimated_volume_from_pressure(target_pressure)
        R0, box_lengths, num_atoms, shift_vectors, atom_per_unitcell = create_supercell(lattice_type, 
                                                                                        volume_per_atom, 
                                                                                        supercell_size, 
                                                                                        supercell_length
                                                                                        )
        print("ensemble: npt (isothermal-isobaric ensemble)") 
        print("target pressure: %.6f (GPa)" % target_pressure)
        if jnp.all(supercell_length > 0):
            print("input initial supercell length:", supercell_length, "(A)")
        print("estimated volume per atom: %.6f (A^3)" % volume_per_atom)
          
    print("isotope:", isotope)
    print("effective mass: %.6f" % effective_mass)
    print("hbar^2/(2m): %.6f (K)" % h2_over_2m)
    print("lattice type:", lattice_type)
    print("number of atoms: %d" % num_atoms)
    print("supercell size:", supercell_size)
    print("supercell lengths:", box_lengths, "(A)")
    print("supercell volume: %.6f (A^3)" % jnp.prod(box_lengths))
    print("temperature: %.6f (K),  beta: %.6f" % (temperature, beta))

####################################################################################################
    print("\n========== Initialize potential energy surface ==========")
    from src.dpjax_lithium import make_dp_model, pkl_mapping, read_pkl_params
    pkl_name = pkl_mapping.get(dpfile)
    dp_energyfn, _ = make_dp_model(pkl_name, 
                                   num_atoms, 
                                   box_lengths_init=box_lengths, 
                                   unit="K"
                                   )
    Vtot0 = dp_energyfn(R0, box_lengths)
    potential_energy = jax.vmap(dp_energyfn, in_axes=(0, None), out_axes=(0)) 

    print("initialize potential energy of", isotope)
    print("load deepmd model", dpfile, "file from:", pkl_name)
    read_pkl_params(pkl_name, verbosity=2)
    
    print("at equilibrium position: V0 = %.6f (K/atom), %.6f (meV/atom)" 
        %(Vtot0/num_atoms, Vtot0/num_atoms*Kelvin_2_meV), flush=True)

####################################################################################################
    print("\n========== Calculate dynamic matrix ==========")
    from src.coordtrans_phon import get_gradient_and_hessian
    V0, Vgrad, Dmat = get_gradient_and_hessian(R0, 
                                               box_lengths, 
                                               effective_mass, 
                                               dp_energyfn, 
                                               hessian_type=hessian_type
                                               )

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
    from src.quantity import pressurefn, stressfn
    
    wsquare_indices, w_indices, Pmat, wkd_rawdata, num_modes \
        = init_coordinate_transformations(coordinate_type, Dmat, shift_vectors, supercell_size, 
                                          atom_per_unitcell, num_atoms, dim, 
                                          cal_kpoint = False, verbosity = 2)

    print("number of modes:", num_modes)
    print("frequency w: (in the unit of meV)\n", w_indices * (1/effective_mass) * Kelvin_2_meV)

    trans_Q2R_novmap, _ = get_coordinate_transforms(num_atoms, dim, coordinate_type)
    trans_Q2R = jax.vmap(trans_Q2R_novmap, in_axes=(0, None, None, None), out_axes=(0))

####################################################################################################
    print("\n========== Initialize orbitals ==========")
    from src.orbitals import get_orbitals_1d, get_orbitals_energy
    sp_orbitals, _ = get_orbitals_1d()
    print("zero-point energy & excited state energies (in meV/atom):")

    for ii in range(num_levels):
        state_idx = jnp.ones((num_modes, ), dtype=jnp.int64) * ii
        state_energy = get_orbitals_energy(state_idx, w_indices)
        print("    level: %.2d    without V0: %.6f    with V0: %.6f" 
            % (ii, 
               (1/effective_mass) * state_energy / num_atoms * Kelvin_2_meV,
               (1/effective_mass) * (state_energy+V0) / num_atoms * Kelvin_2_meV), flush=True)

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
        w_indices_init = w_indices / effective_mass
        if load_ckpt is not None and load_ckpt != "None":
            print("load w_indices_init ...")
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
        print("probability table:  [num_levels: %d,  sampler group: %d]" 
                                    % (num_levels, indices_group))
        print("    #parameters in the probability table: %d" % raveled_params_van.size, flush=True)

        #========== make sampler ==========
        from src.sampler_probtab import make_probtab_sampler
        w_indices_init = w_indices / effective_mass
        if load_ckpt is not None and load_ckpt != "None":
            print("load w_indices_init ...")
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
    print("\n========== Initialize optimizer ==========")

    lr_schedule_class = optax.exponential_decay(init_value       = lr_class,
                                                transition_steps = decay_steps,
                                                decay_rate       = decay_rate,
                                                transition_begin = decay_begin,
                                                end_value        = min_lr_class,
                                                )
    lr_schedule_quant = optax.exponential_decay(init_value       = lr_quant,
                                                transition_steps = decay_steps,
                                                decay_rate       = decay_rate,
                                                transition_begin = decay_begin,
                                                end_value        = min_lr_quant,
                                                )

    optimizer_class = optax.adam(lr_schedule_class)
    optimizer_quant = optax.adam(lr_schedule_quant)
    
    optimizer = optax.multi_transform({'class': optimizer_class, 'quant': optimizer_quant},
                                        param_labels={'van': 'class', 'flw': 'quant'})
    params = {'van': params_van, 'flw': params_flw}
    opt_state = optimizer.init(params)

    print("optimizer adam, learning rate:")
    print("    initial classical: %g    quantum: %g" % (lr_class, lr_quant))
    print("    minimum classical: %g    quantum: %g" % (min_lr_class, min_lr_quant))
    print("    decay rate: %g    decay steps: %d    decay begin: %d" %(decay_rate, decay_steps, decay_begin))
    
    if ensemble == "npt" and relax_steps > 0:
        print("update cell per %d steps, learning rate: %g" % (relax_steps, relax_lr))
        print("    relax_min_lr: %g,  relax_decay: %g" % (relax_min_lr, relax_decay))
        print("    relax_begin: %d,  cal_stress: %d" % (relax_begin, cal_stress))
        print("    relax_steps: %d,  num_recent_vals: %d" % (relax_steps, num_recent_vals))
        print("    relax_therm: %d,  mc_steps: %d,  mc_stddev: %.6f" 
                %(relax_therm, mc_steps, mc_stddev), flush=True)
        
####################################################################################################
    print("\n========== Initialize check point ==========")
    # from src.checkpoint import ckpt_filename, save_pkl_data, load_pkl_data
    mode_str = (f"{isotope}_{dpfile}_{lattice_type}"
        + (f"_npt_p_{target_pressure}" if ensemble == "npt" else "")
        + (f"_nvt_v_{volume_per_atom}" if ensemble == "nvt" else "")
        + f"_n_{num_atoms}"
        + f"_[{''.join(str(x) for x in supercell_size)}]_t_{temperature}")
    
    if van_type == "van_tfhaiku":
        van_str = f"_lev_[{num_levels}_{indices_group}]_van_[{van_layers}_{van_size}_{van_heads}_{van_hidden}]"
    elif van_type == "van_probtab":
        van_str = f"_lev_[{num_levels}_{indices_group}]_van_pt"
        
    if flow_type == "flw_rnvpflax":
        flw_str = f"_flw_[{flow_depth}_{mlp_width}_{mlp_depth}_{flow_st}]"
    elif flow_type == "flw_identity":
        flw_str = f"_flw_[id_{flow_st}]"
    
    mcmc_str = f"_mc_[{mc_steps}_{mc_stddev}]"
    
    opt_str = f"_lr_[{lr_class}_{lr_quant}_{min_lr_class}_{min_lr_quant}_{decay_rate}_{decay_steps}]"

    if relax_steps > 0 and ensemble == "npt":
        rlx_str = f"_rlx_[{relax_lr}_{relax_min_lr}_{relax_decay}_{relax_steps}_{relax_therm}_{num_recent_vals}]"
    else:
        rlx_str = ""
        
    bth_str = f"_bth_[{batch}_{acc_steps}]_key_{args.seed}"

    path = (folder + mode_str + van_str + flw_str + "_" + coordinate_type 
            + ("_hut" if hutchinson else "") + mcmc_str + opt_str + rlx_str + bth_str)

    print("#file path:", path)
    if not os.path.isdir(path):
        os.makedirs(path)
        print("#create path: %s" % path)

    from src.vmc import sample_stateindices_and_x, make_loss, \
                        update_van_flw, update_cell, calculate_means_and_stds, \
                        store_recent_stress, calculate_recent_stress
                
    #========== load optimizer & params ==========
    if load_ckpt is not None and load_ckpt != "None":
        # print("load pkl data from:\n", load_ckpt)
        # ckpt = load_pkl_data(load_ckpt)
        # opt_state = ckpt["opt_state"]
        print("load params_van and params_flw ...", flush=True)
        params_van, params_flw = ckpt["params_van"], ckpt["params_flw"]
    else:
        print("No checkpoint file found. Start from scratch.")
    params_van, params_flw = replicate((params_van, params_flw), num_devices)

    #========== load coordinates and sampler ==========
    if load_ckpt is not None and load_ckpt != "None" \
    and ckpt["x"].size == num_devices * batch_per_device * num_modes:
        x = ckpt["x"].reshape(num_devices, batch_per_device, num_modes, 1)
    else:
        x = 0.1 * jax.random.normal(key, (num_devices, batch_per_device, num_modes, 1), dtype=jnp.float64)
    keys = jax.random.split(key, num_devices)
    x, keys = shard(x), shard(keys)
    print("keys shape:", keys.shape)
    print("x.shape:", x.shape)

    #========== thermalized ==========
    for ii in range(mc_therm):
        t1 = time.time()
        keys, state_indices, x, accept_rate = sample_stateindices_and_x(keys, 
                                                sampler, params_van,
                                                logp, x, params_flw, w_indices,
                                                mc_steps, mc_stddev, index_list)
        t2 = time.time()
        accept_rate = jnp.mean(accept_rate)
        print("---- thermal step: %d,  ac: %.4f,  dx: %.4f,  dt: %.3f ----" 
        % (ii+1, accept_rate, mc_stddev, t2-t1), flush=True)
        mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)

    #========== observable, loss function ==========
    observable_and_lossfn = make_loss(effective_mass, beta, index_list, clip_factor,
                                      cal_stress, pressurefn, stressfn, dp_energyfn, 
                                      log_prob, logpsi, logpsi_grad_laplacian,
                                      potential_energy, trans_Q2R)

    #========== open file ==========
    log_filename = os.path.join(path, "data.txt")
    print("#data name: ", log_filename, flush=True)
    f = open(log_filename, "w", buffering=1, newline="\n")

####################################################################################################
    print("\n========== Training ==========")
    #========== circulate ==========
    t0 = time.time()
    recent_T_vals, recent_P_vals = None, None
    print("start training:")
    print("    F, E, K, V, G (in the unit of meV/atom)")
    print("    S (entropy per atom)")
    print("    P, T (pressure and stress tensor in the unit of GPa)")

    for ii in range(1, epoch+1):
        
        ####========== structural relaxation (loss: Gibbs free energy) ==========
        if relax_steps == 0 or ensemble == "nvt":  
            pass
        
        elif ii % relax_steps == 1 and ii > relax_begin and cal_stress > 0 and ensemble == "npt":
            tc1 = time.time()
            recent_T, recent_T_std, recent_P, recent_P_std \
                = calculate_recent_stress(recent_T_vals, recent_P_vals, num_recent_vals)
            
            box_lengths_old, box_lengths, R0, wsquare_indices, w_indices, Pmat \
                = update_cell(box_lengths, R0, relax_lr, recent_T, target_pressure, 
                              lattice_type, effective_mass, hessian_type, coordinate_type, 
                              shift_vectors, supercell_size, atom_per_unitcell, 
                              get_gradient_and_hessian, init_coordinate_transformations, 
                              dp_energyfn
                              )
            tc2 = time.time()
            
            print("**relax**  box: [%.16f %.16f %.16f] vol: %.16f -> box: [%.16f %.16f %.16f] vol: %.16f"
                %(*tuple(box_lengths_old), jnp.prod(box_lengths_old)/num_atoms,
                  *tuple(box_lengths),     jnp.prod(box_lengths)/num_atoms),
                " recentP: %.4f (%.4f)  recentT: %.4f %.4f %.4f (%.4f %.4f %.4f)  lr: %.6f  dt: %.3f" 
                %(recent_P,         recent_P_std, 
                  *tuple(recent_T), *tuple(recent_T_std), 
                  relax_lr, 
                  tc2-tc1
                  ), 
                 flush=True
                 )
            
            relax_lr = relax_lr * relax_decay
            if relax_lr < relax_min_lr: 
                relax_lr = relax_min_lr
            args.relax_lr = relax_lr
            
            # thermalize after relaxation
            tt1 = time.time()
            for kk in range(relax_therm):
                keys, state_indices, x, accept_rate = sample_stateindices_and_x(keys, 
                                                        sampler, params_van,
                                                        logp, x, params_flw, w_indices,
                                                        mc_steps, mc_stddev, index_list)
                accept_rate = jnp.mean(accept_rate)
                mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)
            tt2 = time.time()
            print("**relax**  thermal for %d steps:  ac: %.4f,  dx: %.4f,  dt: %.3f" 
                % (kk+1, accept_rate, mc_stddev, tt2-tt1), 
                flush=True
                )
        
        ####========== update van, flw and physical quantities  (loss: Helmholtz free energy) ==========
        tf1 = time.time()
        datas_acc = replicate({"F_mean": 0., "F2_mean": 0., # Helmholtz free energy
                               "E_mean": 0., "E2_mean": 0., # local energy
                               "K_mean": 0., "K2_mean": 0., # kinetic energy
                               "V_mean": 0., "V2_mean": 0., # potential energy
                               "S_mean": 0., "S2_mean": 0., # entropy
                               "G_mean": 0., "G2_mean": 0., # Gibbs free energy
                               "P_mean": 0., "P2_mean": 0., # pressure
                               "T_mean": jnp.zeros(3), "T2_mean": jnp.zeros(3), # stress
                               }, 
                              num_devices
                              )
        grads_acc = shard( jax.tree.map(jnp.zeros_like, {'van': params_van, 'flw': params_flw}))
        class_score_acc = shard(jax.tree.map(jnp.zeros_like, params_van))
        quant_score_acc = shard(jax.tree.map(jnp.zeros_like, params_flw))
        accept_rate_acc = shard(jnp.zeros(num_devices))
        x_epoch = jnp.zeros((acc_steps, num_devices, batch_per_device, num_modes, 1), 
                            dtype=jnp.float64)
        
        for acc in range(acc_steps):   
            keys, state_indices, x, accept_rate = sample_stateindices_and_x(keys, 
                                                            sampler, params_van,
                                                            logp, x, params_flw, w_indices,
                                                            mc_steps, mc_stddev, index_list)

            accept_rate_acc += accept_rate
            final_step = (acc == (acc_steps-1))
            
            params_van, params_flw, opt_state, datas_acc, grads_acc, class_score_acc, quant_score_acc \
                    = update_van_flw(params_van, params_flw, opt_state, 
                                    state_indices, x, w_indices, box_lengths, R0, Pmat, keys, 
                                    datas_acc, grads_acc, class_score_acc, quant_score_acc, 
                                    acc_steps, final_step, observable_and_lossfn, optimizer)
            
            x_epoch = x_epoch.at[acc, :].set(x)
        
        # data are in the unit of K/atom or K/A^3
        data = jax.tree.map(lambda x: x[0], datas_acc)
        accept_rate = accept_rate_acc[0] / acc_steps
        mc_stddev = automatic_mcstddev(mc_stddev, accept_rate)
        args.mcstddev = mc_stddev
        # change the unit from K into meV/atom and GPa
        computed_quantities = calculate_means_and_stds(data, num_atoms, batch, acc_steps)
        F, F_std = computed_quantities["F"] # Helmholtz free energy
        E, E_std = computed_quantities["E"] # energy
        K, K_std = computed_quantities["K"] # kinetic energy
        V, V_std = computed_quantities["V"] # potential energy
        S, S_std = computed_quantities["S"] # entropy
        G, G_std = computed_quantities["G"] # Gibbs free energy
        P, P_std = computed_quantities["P"] # pressure
        T, T_std = computed_quantities["T"] # stress
        
        current_lr_class = lr_schedule_class(ii)
        current_lr_quant = lr_schedule_quant(ii)
        
        if cal_stress > 0: # calculate stress for npt ensemble
            recent_T_vals, recent_P_vals = store_recent_stress(recent_T_vals, 
                                                               recent_P_vals, 
                                                               computed_quantities["T"], 
                                                               computed_quantities["P"], 
                                                               num_recent_vals
                                                               )
            ####========== print ==========
            tf2 = time.time()
            print("iter: %05d" % ii,
            " F: %.3f (%.3f)  E: %.3f (%.3f)  K: %.3f (%.3f)  V: %.3f (%.3f)  S: %.6f (%.6f)  \
G: %.3f (%.3f)  P: %.4f (%.4f)  T: %.4f %.4f %.4f (%.4f %.4f %.4f)"
            % (F, F_std, 
               E, E_std, 
               K, K_std, 
               V, V_std, 
               S, S_std,
               G, G_std, 
               P, P_std, 
               *tuple(T), *tuple(T_std)
               ),
            " ac: %.4f  dx: %.6f  lr: %.10f  %.10f  dt: %.3f" 
            % (accept_rate, 
               mc_stddev, 
               current_lr_class, 
               current_lr_quant, 
               tf2-tf1
               ), 
            flush=True
            )
            ####========== save txt data ==========
            f.write( ("%6d" + "  %.12f"*14 + "  %.16f"*9 + "  %.16f"*2 + "  %.16f"*2 + "\n") 
                    % (ii, 
                       F, F_std, 
                       E, E_std, 
                       K, K_std, 
                       V, V_std, 
                       S, S_std, 
                       G, G_std, 
                       P, P_std, 
                       *tuple(T), *tuple(T_std), 
                       *tuple(box_lengths), 
                       accept_rate, 
                       mc_stddev, 
                       current_lr_class, 
                       current_lr_quant
                       )
                    )
        
        else: # no stress calculation for nvt ensemble
            ####========== print ==========
            tf2 = time.time()
            print("iter: %05d" % ii,
            " F: %.3f (%.3f)  E: %.3f (%.3f)  K: %.3f (%.3f)  V: %.3f (%.3f)  S: %.6f (%.6f)  \
G: %.3f (%.3f)  P: %.4f (%.4f)"
            % (F, F_std, 
               E, E_std, 
               K, K_std, 
               V, V_std, 
               S, S_std, 
               G, G_std, 
               P, P_std
               ),
            " ac: %.4f  dx: %.6f  lr: %.10f  %.10f  dt: %.3f" 
            % (accept_rate, 
               mc_stddev, 
               current_lr_class, 
               current_lr_quant, 
               tf2-tf1
               ), 
            flush=True
            )
            ####========== save txt data ==========
            f.write( ("%6d" + "  %.12f"*14 + "  %.16f"*2 + "  %.16f"*2 + "\n") 
                    % (ii, 
                       F, F_std, 
                       E, E_std, 
                       K, K_std, 
                       V, V_std, 
                       S, S_std, 
                       G, G_std, 
                       P, P_std, 
                       accept_rate, 
                       mc_stddev, 
                       current_lr_class, 
                       current_lr_quant
                       )
                    )
            
        ####========== save ckpt data ==========
        if ii % ckpt_epochs == 0:
            ckpt = {"keys": keys, 
                    "x": x, 
                    "x_epoch": x_epoch, 
                    "opt_state": opt_state,
                    "params_flw": jax.tree.map(lambda x: x[0], params_flw), 
                    "params_van": jax.tree.map(lambda x: x[0], params_van),
                    "args": args, 
                    "num_atoms": num_atoms, 
                    "R0": R0, 
                    "box_lengths": box_lengths, 
                    "shift_vectors": shift_vectors, 
                    "atom_per_unitcell": atom_per_unitcell, 
                    "num_modes": num_modes,
                    "w_indices": w_indices, 
                    "wsquare_indices": wsquare_indices,
                    "Pmat": Pmat, 
                    "index_list": index_list,
                    "w_indices_init": w_indices_init, 
                    "num_devices": num_devices,
                    }
            
            save_ckpt_filename = ckpt_filename(ii, path)
            save_pkl_data(ckpt, save_ckpt_filename)
            print("save file: %s" % save_ckpt_filename, flush=True)
            print("total time used: %.3fs (%.3fh),  training speed: %.3f epochs per hour. (%.3fs per step)" 
                % ((tf2-t0), 
                   (tf2-t0)/3600, 
                   3600.0/(tf2-t0)*ii, 
                   (tf2-t0)/ii
                   ), 
                flush=True
                )
        
    f.close()
