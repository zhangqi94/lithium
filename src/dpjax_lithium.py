import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import itertools
import json
import pickle

diag_eps = 1e-10
##############################################################
"""
    ** Deep model (jax version) for lithium **

    Ref: J. Chem. Phys. 159, 054801 (2003)
    DeePMD-kit v2: A software package for deep potential models

    convert a deep potential model from deepmd+tensorflow to jax
    imitate from:
    https://code.itp.ac.cn/wanglei/hydrogen/-/blob/dp-base-wanghan-wtb/src/dp_jax.py
"""
##############################################################
##========== apply layer of neural network ==========

def apply_layer(xx, ww, bb, activation=True, idt=None):
    if xx.shape[-1] == bb.shape[0] // 2:
        addon = jnp.concatenate([xx, xx], axis=-1)
    elif xx.shape[-1] == bb.shape[0]:
        addon = xx
    else:
        addon = None
    yy = jnp.matmul(xx, ww) + bb
    if activation:
        yy = jnp.tanh(yy)
    if idt is not None:
        yy = idt * yy
    if addon is not None:
        yy = addon + yy
    return yy

##############################################################
##========== switch function s(r) (only tanh is available) ==========
def switch_func_tanh(xx, rc_cntr = 3.0, rc_sprd = 0.2):
    uu = (xx - rc_cntr) / rc_sprd
    return 0.5 * (1. - jnp.tanh(uu))

def spline_func(xx, rc, rc_smth):
    uu = (xx - rc_smth) / (rc - rc_smth)
    return uu*uu*uu * (-6 * uu*uu + 15 * uu - 10) + 1

def switch_func_poly(xx, rc = 6.0, rc_smth = 0.0):
    ret = \
        1.0 * (xx < rc_smth) + \
        spline_func(xx, rc, rc_smth) * jnp.logical_and(xx >= rc_smth, xx < rc) + \
        0.0 * (xx >= rc)
    return ret

def choose_swf(rc_mode = 'poly', rc_cntr = 6.0, rc_sprd = 0.0):
    do_smth = \
        (rc_mode is not None) and (rc_mode != 'none') and \
        (rc_cntr is not None) and (rc_sprd is not None)
    if do_smth:
        if rc_mode == 'tanh':
            swf = lambda xx: switch_func_tanh(xx, rc_cntr, rc_sprd)
        elif rc_mode == 'poly':
            swf = lambda xx: switch_func_poly(xx, rc_cntr, rc_sprd)
        else:
            raise RuntimeError('unknown rc mode', rc_mode)
    else:
        swf = jnp.ones_like
    return swf

##############################################################
##========== comput ncopy ==========
def compute_ncopy(
        rc : float,
        lattice : jnp.array,
        rc_shell : float = 1e-5,
) -> jnp.array:
    vol = jnp.linalg.det(lattice)
    tofacedist = jnp.cross(lattice[[1,2,0],:], lattice[[2,0,1],:])
    tofacedist = vol * jnp.reciprocal(jnp.linalg.norm(tofacedist, axis=1))
    ncopy = (rc+rc_shell) * jnp.reciprocal(tofacedist)
    ncopy = jnp.array(jnp.ceil(ncopy), dtype=int)    
    return ncopy


def compute_copy_idx(
        rc : float,
        lattice : jnp.array,
):
    ncopy = compute_ncopy(rc, lattice)
    ordinals = jnp.asarray(list(
        itertools.product(
            jnp.arange(-ncopy[0], ncopy[0]+1),
            jnp.arange(-ncopy[1], ncopy[1]+1),
            jnp.arange(-ncopy[2], ncopy[2]+1),
        )))
    ordinals = jnp.asarray(sorted(ordinals, key=jnp.linalg.norm))
    return ordinals


def compute_shift_vec(
        rc : float,
        lattice : jnp.array,
) -> jnp.array:
    ordinals = compute_copy_idx(rc, lattice)
    return jnp.matmul(ordinals, lattice)


def compute_background_coords(
        xx : jnp.array,
        lattice : jnp.array,
        shift_vec : jnp.array,
):
    ss = shift_vec
    xx = jnp.reshape(xx, [1,-1,3])
    ss = jnp.reshape(ss, [-1,1,3])
    coord = xx[None,:,:] + shift_vec[:,None,:]
    return coord

##========== comput rij ==========
def compute_rij(
        xx : jnp.array,
        lattice : jnp.array = None,
        shift_vec : jnp.array = None,
):
    return compute_rij_2(xx, xx, lattice, shift_vec)


def compute_rij_2(
        cc : jnp.array,
        yy : jnp.array,
        lattice : jnp.array = None,
        shift_vec : jnp.array = None,
):
    if lattice is not None:
        bk_yy = compute_background_coords(yy, lattice, shift_vec).reshape([-1,3])
    else:
        bk_yy = yy
    # np0 x np1 x 3
    rij = - bk_yy[None,:,:] + cc[:,None,:]    
    return rij

##############################################################
##========== compute env mat ==========
def _env_mat_rinv(
        rij,
        power : float = 1.0,
        rinv_shift : float = 1.0,
        switch_func = None,
):
    """
    env mat constructed as 

        1            xij            yij            zij
    ---------,  -------------, -------------, -------------, 
    rij^p + s   rij^(p+1) + s  rij^(p+1) + s  rij^(p+1) + s

    p is given by power
    s is given by rinv_shift
    
    for p = 1, s = 0:
    1 / rij, s(|rij|) xij / rij^2, s(|rij|) yij / rij^2, s(|rij|) zij / rij^2
    """
    if power != 1.0:
        raise RuntimeError(f'the power {power} is not supported. only allows power==1.')
    # np0 x np1 x 3
    np0 = rij.shape[0]
    np1 = rij.shape[1]
    # np0 x np1
    nrij = jnp.linalg.norm(rij, axis=2)
    inv_nrij = switch_func(nrij)/(nrij + rinv_shift)
    # flaten 
    trij = jnp.reshape(rij, [-1,3])
    tnrij = jnp.tile(
                jnp.reshape(switch_func(nrij)/(nrij*nrij + rinv_shift), [-1, 1],),
                [1,3])
    # np0 x np1 x 3
    env_mat = jnp.reshape(jnp.multiply(trij, tnrij), [np0, np1, 3])
    # np0 x np1 x 4
    env_mat = jnp.concatenate(
        (jnp.reshape(inv_nrij, [np0, np1, 1]), env_mat),
        axis = 2)
    # np0 x np1 x 4, np0 x np1, np0 x np1
    return env_mat, inv_nrij, nrij

def _env_mat(
        xx,
        power : float = 1.0,
        rinv_shift : float = 1.0,
        rc_mode : str = 'none', # 'tanh' or 'poly' or 'none'
        rc_cntr : float = None,
        rc_sprd : float = None,
        cut_dim : int = None,
        lattice : jnp.array = None,
        shift_vec : jnp.array = None,
):
    # np0 x 3
    xx = jnp.reshape(xx, [-1, 3])
    rij = compute_rij(xx, lattice, shift_vec)
    # np0 x np1 x 3
    diag_shift = diag_eps * jnp.tile(jnp.expand_dims(jnp.eye(rij.shape[0], rij.shape[1]), axis=2), [1,1,3])
    rij += diag_shift
    switch_func = choose_swf(rc_mode, rc_cntr, rc_sprd)
    # np0 x np1 x 4, np0 x np1, np0 x np1
    env_mat, inv_nrij, nrij = _env_mat_rinv(
        rij, 
        power=power, 
        rinv_shift=rinv_shift,
        switch_func=switch_func,
    )
    if cut_dim is not None:
        # cut_dim(npart/nele) x npart x 4
        env_mat = jax.lax.dynamic_slice(env_mat, (0,0,0), (cut_dim, env_mat.shape[1], env_mat.shape[2]))
        nrij = jax.lax.dynamic_slice(nrij, (0,0), (cut_dim, nrij.shape[1]))
    # np0 x np1 x 4, np0 x np1, np0 x np1
    return env_mat, inv_nrij, nrij


def sort_env_mat(env_mat,):
    idx = jnp.argsort(env_mat[...,0])
    return env_mat[...,idx[::-1],:]

batch_sort_env_mat = jax.vmap(sort_env_mat, in_axes=(0))

def compute_env_mat(
        xx,
        nsel,
        power : float = 1.0,
        lattice : jnp.array = None,
        shift_vec : jnp.ndarray = None,
        rinv_shift : float = 0.0,
        rc : float = None,
        rc_smth : float = None,
):
    # natom x (natom x nshift) x 4
    env_mat, inv_nrij, nrij = _env_mat(xx, power, rinv_shift, "poly", rc, rc_smth, None, lattice, shift_vec)
    # natom x (natom x nshift) x 4
    env_mat = batch_sort_env_mat(env_mat)
    env_mat = env_mat[:,1:,:]
    # natom x nsel x 4
    natom, nans, _ = env_mat.shape
    if nsel > nans:
        ret = jnp.concatenate([
            env_mat,
            jnp.zeros([natom, nsel - nans, 4])
            ], axis = 1)
    elif nsel < nans:
        ret = env_mat[:,:nsel,:]
    return ret

####################################################################################################
####################################################################################################
## make deepmd model
def make_dp_model(pkl_name: str, 
                  natom: int, 
                  box_lengths_init: jnp.array = None, 
                  unit: str = "K",
                  ):
    """
    Inputs: 
        - pkl_name: name of the pickle (numpy) file of the dpmodel
        - natom: number of atoms
        - box_lengths_init: initialize box lengths 
            (only to calculate the shift vector in dp_energy_withboxfn mode)
        - unit: unit of the energy (either "eV" or "K")
        
    Outputs: 
        - dp_energyfn: function to calculate energy
        - dp_forcefn: function to calculate force
        - dp_energy_withboxfn: function to calculate energy with various box lengths
    """

    cell_init = jnp.diag(box_lengths_init)
    with open(pkl_name, 'rb') as f:
        dpnp = pickle.load(f)
    
    metadata, params = dpnp["metadata"], dpnp["params"]

    ## initial cell size calculate shift_vec
    shift_vec0_e2 = jnp.asarray(
                compute_shift_vec(rc=metadata["se_e2_a"]["rcut"], lattice=cell_init)
                ) / box_lengths_init
    shift_vec0_e3 = jnp.asarray(
                compute_shift_vec(rc=metadata["se_e3"]["rcut"],   lattice=cell_init)
                ) / box_lengths_init
    ##========== apply env_mat ==========
    def apply_env_mat_e2(coord, box_lengths, nsel, rc, rc_smth=0.0):
        # natom, nsel = 32, 200; rc, rc_smth = 6.0, 0.0
        # assumes all the types are 0
        coord = coord.reshape([-1,3])
        env_mat = compute_env_mat(coord, 
                                  nsel, 
                                  lattice=jnp.diag(box_lengths), 
                                  shift_vec=shift_vec0_e2*box_lengths, 
                                  rc=rc, 
                                  rc_smth=rc_smth)
        avg = params["em2_avgstd"][0].reshape([-1,4])
        std = params["em2_avgstd"][1].reshape([-1,4])
        env_mat = (env_mat - avg[None,...]) / std[None,...]
        return env_mat

    def apply_env_mat_e3(coord, box_lengths, nsel, rc, rc_smth=0.0):
        # natom, nsel = 32, 25; rc, rc_smth = 3.0, 0.0
        # assumes all the types are 0
        coord = coord.reshape([-1,3])
        env_mat = compute_env_mat(coord, 
                                  nsel, 
                                  lattice=jnp.diag(box_lengths), 
                                  shift_vec=shift_vec0_e3*box_lengths, 
                                  rc=rc, 
                                  rc_smth=rc_smth)
        avg = params["em3_avgstd"][0].reshape([-1,4])
        std = params["em3_avgstd"][1].reshape([-1,4])
        env_mat = (env_mat - avg[None,...]) / std[None,...]
        return env_mat

    ##========== apply embedding ==========
    def apply_embedding_e2(params, xx):
        nlayer = len(params['se_e2_a'])
        for ii in range(nlayer):
            xx = apply_layer(xx, *params['se_e2_a'][ii], activation=True)
        return xx

    def apply_embedding_e3(params, xx):
        nlayer = len(params['se_e3'])
        for ii in range(nlayer):
            xx = apply_layer(xx, *params['se_e3'][ii], activation=True)
        return xx

    ##========== apply fitting ==========
    def apply_fitting(params, xx):
        nlayer = len(params["fit"])
        for ii in range(nlayer):
            if ii == 0:
                idt = None
            else:
                idt = params["fit_idt"][ii-1]
            xx = apply_layer(xx, *params["fit"][ii], activation=True, idt=idt)
        xx = apply_layer(xx, *params["fit_final"], activation=False)
        return xx    
    
    ##========== apply fitting ==========
    def get_dd2(coord, box_lengths, nsel=200, rc=6.0, rc_smth=0.0, axis_neuron=16):
        '''
            Two-body embedding DeepPot-SE
            J. Chem. Phys. 159, 054801 (2003)
            Eq. (11): Di = (1/Nc^2) Gi^T Ri Ri^T Gi<
            Eq. (14): (Gi)j = NN_e2(s(rij))
            
            rr_i:   Ri,    shape = (natom, nsel, 4)
            gg_i:   Gi,    shape = (natom, nsel, M)
            rg:    (1/Nc) * Ri Gi^T,    shape = (natom, M,  4)
            rg2:   (1/Nc) * Ri Gi<^T,   shape = (natom, M2, 4)
            dd2:    Di,    shape = (natom, M*M2)
        '''
        #axis_neuron = 16
        
        # rr_i: natom x nsel x 4     e.g.(32, 200, 4)
        rr_i = apply_env_mat_e2(coord, box_lengths, nsel, rc, rc_smth)
        # gg_i: natom x nsel x M     e.g.(32, 200, 80)
        gg_i = apply_embedding_e2(params, rr_i[..., 0:1])
        # rg: natom x M x 4     e.g.(32, 80, 4)
        rg = jnp.einsum('nmi,nmj->nji', rr_i, gg_i) / jnp.float64(nsel)
        # rg2: natom x M2 x 4     e.g.(32, 16, 4)
        rg2 = rg[:, :axis_neuron, :]
        # dd2: natom x M x M2     e.g.(32, 80, 16)
        dd2 = jnp.einsum('nid,njd->nij', rg, rg2)
        # dd2: natom x (M x M2)     e.g.(32, 1280)
        dd2 = dd2.reshape([natom, -1])
        return dd2

    def get_dd3(coord, box_lengths, nsel=25, rc=3.0, rc_smth=0.0, ng=16):
        '''
            Three-body embedding DeepPot-SE
            J. Chem. Phys. 159, 054801 (2003)
            Eq. (15): Di = (1/Nc^2) (Ri Ri^T) : Gi
                    where ':' indicates the contraction between matrix (Ri Ri^T) 
                    and the first two dimensions of tensor Gi.
            Eq. (16): (Gi)jk = NN_e3((theta_i)jk)
                    where (theta_i)jk = (Ri)j (Ri)k
        '''
        #ng = 16 ## neural network output size

        # rr: natom x nsel x 4    e.g.(32, 25, 4)
        rr = apply_env_mat_e3(coord, box_lengths, nsel, rc, rc_smth)
        
        ## rr_i: natom x nsel^2 x 3    e.g.(32, 625, 3)
        ## rr_j: natom x nsel^2 x 3    e.g.(32, 625, 3)
        ti, tj = zip(*itertools.product(jnp.arange(nsel), repeat=2))
        rr_i = rr[:, ti, 1:]
        rr_j = rr[:, tj, 1:]

        ## env_ij: natom x nsel^2    e.g.(32, 625)
        ## env_ij_reshape: natom x nsel^2 x 1    e.g.(32, 625, 1)
        env_ij = jnp.einsum("ijm,ijm->ij", rr_i, rr_j)
        env_ij_reshape = env_ij[:, :, None]
        ## gg: natom x nsel^2 x ng    e.g.(32, 625, 16)
        gg = apply_embedding_e3(params, env_ij_reshape)
        ## dd3: natom x ng    e.g.(32, 16)
        dd3 = jnp.einsum("ij,ijm->im", env_ij, gg) / jnp.float64(nsel**2)
        return dd3
    
    ## Get energy in the unit of Kelvin or eV  (Boltzmann constant)
    if unit == "K":
        kb_eV_2_Kelvin = 1 / 8.617333262145e-5
    elif unit == "eV":
        kb_eV_2_Kelvin = 1.0
    
    ## Get energy with various coordinates and box lengths
    ## in order to calculate pressure p=dU/dV (see quantity.py)
    def dp_energyfn(coord, box_lengths):
        coord = coord - box_lengths * jnp.floor(coord / box_lengths)
        dd2 = get_dd2(coord, 
                      box_lengths, 
                      nsel        = metadata["se_e2_a"]["nsel"], 
                      rc          = metadata["se_e2_a"]["rcut"], 
                      rc_smth     = metadata["se_e2_a"]["rcut_smth"], 
                      axis_neuron = metadata["se_e2_a"]["axis_neuron"]
                      )
        dd3 = get_dd3(coord, 
                      box_lengths, 
                      nsel    = metadata["se_e3"]["nsel"], 
                      rc      = metadata["se_e3"]["rcut"], 
                      rc_smth = metadata["se_e3"]["rcut_smth"], 
                      ng      = metadata["se_e3"]["ng"]
                      )
        dd = jnp.concatenate((dd2, dd3), axis = -1)
        energy = apply_fitting(params, dd)
        return jnp.sum(energy) * kb_eV_2_Kelvin
    
    ## Get force in the unit of Kelvin/Angstrom or eV/Angstrom
    def dp_forcefn(coord, box_lengths):
        def energyfn_without_box(coord):
            return dp_energyfn(coord, box_lengths)
        grad_energyfn = jax.grad(energyfn_without_box)
        force = -1.0 * grad_energyfn(coord)
        return force

    return dp_energyfn, dp_forcefn

####################################################################################################
pkl_mapping = {
    "dp0": "./src/dpmodel/fznp_dp0.pkl",
    "dp2": "./src/dpmodel/fznp_dp2.pkl",
    "dp3": "./src/dpmodel/fznp_dp3.pkl",
    "dp4": "./src/dpmodel/fznp_dp4.pkl"
    }

####################################################################################################
def read_pkl_params(pkl_name, 
                    verbosity=0
                    ):
    
    from pprint import pprint
    
    # def count params function
    def count_params(data):
        total_count = 0
        if isinstance(data, dict):
            for key, value in data.items():
                total_count += count_params(value)
        elif isinstance(data, list):
            for item in data:
                total_count += count_params(item)
        elif isinstance(data, np.ndarray):
            total_count += data.size
        return total_count

    # load file
    with open(pkl_name, 'rb') as f:
        dpnp = pickle.load(f)
    metadata, params = dpnp["metadata"], dpnp["params"]
    
    num_params = count_params(params)
    num_params_e2 = count_params(params['se_e2_a'])
    num_params_e3 = count_params(params['se_e3'])
    num_params_fit = count_params(params['fit'])
    
    if verbosity >= 2:
        print("# metadata:")
        pprint(metadata)
        
    if verbosity >= 1:
        print(f"# numparams: {num_params} [{num_params_e2}, {num_params_e3}, {num_params_fit}]")

    return metadata, num_params, num_params_e2, num_params_e3, num_params_fit
    
####################################################################################################
if __name__ == "__main__":
    print("======== test dp-jax model of lithium ========")    
    num_atoms = 32
    dim, L = 3, 10
    box_lengths = jnp.array([L, L, L])
    
    np.random.seed(42)
    coord = jnp.array( np.random.uniform(0., L, (num_atoms, dim)) )
    print("num_atoms:", num_atoms)
    print("box_lengths:", box_lengths)
    print("coordinate:", coord)
    
    def test_dp_jax(dpfile):
        pkl_name = pkl_mapping.get(dpfile)

        read_pkl_params(pkl_name, 
                        verbosity=2
                        )

        dp_energyfn, dp_forcefn = make_dp_model(pkl_name, 
                                                num_atoms, 
                                                box_lengths_init=box_lengths, 
                                                unit="eV"
                                                )

        # energy = jax.jit(dp_energyfn)(coord)
        # print("energy per atom (with jit): ", energy/num_atoms) 
        energy = jax.jit(dp_energyfn)(coord, box_lengths)
        print("energy per atom (with box lengths): ", energy/num_atoms)
        #force = jax.jit(dp_forcefn)(coord, box_lengths)
        #print("force (with jit): ", force) 
        return 

    print("========== frozen_model dp0 ==========")
    test_dp_jax("dp0")
    print("========== frozen_model dp2 ==========")
    test_dp_jax("dp2")
    print("========== frozen_model dp3 ==========")
    test_dp_jax("dp3")
    print("========== frozen_model dp4 ==========")
    test_dp_jax("dp4")

####################################################################################################
'''
cd /home/zhangqi/MLCodes/lithium/
cd /mnt/t02home/MLCodes/lithium/
python3 src/dpjax_lithium.py
'''
