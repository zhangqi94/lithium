import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

####################################################################################################

@partial(jax.jit, static_argnums=(2, 3, 4))
def get_gradient_and_hessian(R0, 
                             box_lengths, 
                             effective_mass,
                             energyfn, 
                             hessian_type="jax"):
    """
    Calculate the potential energy of the system, and optionally its gradient and Hessian matrix.
    Inputs:
        - R0 : ndarray, shape=(N, 3)
            The initial (equilibrium) Cartesian coordinates of atoms in the supercell.   
        - box_lengths : ndarray, shape=(3,)
            The lengths of the simulation box in each dimension, given in angstroms (A).
        - effective_mass : float
            A scaling factor representing the effective mass.
        - energyfn : function
            A function that computes the energy with atomic coordinates and box lengths.
        - hessian_type : str, optional, default="jax"
            The method used to calculate the Hessian matrix. 
            Options are "jax", "for1", and "for2". (foriloop for "row" and "both row and column")
    Outputs:
        - V0 : float
            The potential energy at the equilibrium positions.
        - Vgrad : ndarray, shape=(N-1)*3, optional
            The gradient of the potential energy at the equilibrium positions, 
            returned if `grad_and_hessian=True`.
        - Dmat : ndarray, shape=((N-1)*3, (N-1)*3), optional
            The Hessian matrix (dynamic matrix) of the potential energy at the equilibrium 
            positions, returned if `grad_and_hessian=True`.
    """

    num_atoms, dim = R0.shape
    event_shape = num_atoms * dim
    R0_flatten = R0.flatten()

    def energyfn_without_box(coord):
        return energyfn(coord, box_lengths)

    def potential_R_flatten(R_flatten):
        R = R_flatten.reshape(num_atoms, dim)
        V = energyfn_without_box(R) * effective_mass
        return V
    
    # Calculate the potential energy at the equilibrium positions
    V0 = potential_R_flatten(R0_flatten)
    
    # ========== Calculate gradient of potential energy ==========
    potential_R_grad = jax.jacrev(potential_R_flatten)
    Vgrad = potential_R_grad(R0_flatten)
    
    # ========== Calculate Hessian matrix (dynamic matrix) ==========
    # Use JAX's automatic differentiation to calculate the full Hessian matrix
    if hessian_type == "jax":
        potential_R_hessian = jax.hessian(potential_R_flatten)
        Dmat = potential_R_hessian(R0_flatten)
        
    # Calculate Hessian matrix row by row using a fori_loop
    elif hessian_type == "for1":
        Dmat = jnp.zeros((event_shape, event_shape), dtype = jnp.float64)
        # Calculate a single row of the Hessian matrix 
        def hessian_row(x, row_idx):
            def partial_grad(x):
                return potential_R_grad(x)[row_idx]
            return jax.jacrev(partial_grad)(x)
            # return jax.jacfwd(partial_grad)(x) #(may run out of memory)
        def body_fun(i, Dmat):
            Dmat = Dmat.at[i, :].set(hessian_row(R0_flatten, i))
            return Dmat
        Dmat = jax.lax.fori_loop(0, event_shape, body_fun, Dmat)
    
    # Calculate Hessian matrix element by element using a fori_loop
    elif hessian_type == "for2":
        Dmat = jnp.zeros((event_shape, event_shape), dtype = jnp.float64)        
        # Calculate a single element of the Hessian matrix
        def hessian_fun(x, idx):
            i, j = idx//event_shape, idx%event_shape
            def partial_grad(x):
                return potential_R_grad(x)[i]
            return jax.jacrev(partial_grad)(x)[j]
            # return jax.jacfwd(partial_grad)(x)[j]
        def body_fun(idx, Dmat):
            i, j = idx//event_shape, idx%event_shape
            Dmat = Dmat.at[i, j].set(hessian_fun(R0_flatten, idx))
            return Dmat
        Dmat = jax.lax.fori_loop(0, event_shape**2, body_fun, Dmat)       
            
    return V0, Vgrad, Dmat

####################################################################################################
## translate into phonon coordiante
# @partial(jax.jit)
def get_Dmateigs(Dmat, 
                 tolerance = 1e-4,
                 verbosity = 0
                 ):
    """
        Compute the eigenvalues and eigenvectors of a dynamic matrix (Dmat), and sort zero 
        frequencies to the top. (it can not jax.jit!!!)
        
        Mathematical operations:
            D @ P = P @ W        
            D = P @ W @ P.T
            W = P.T @ W @ P
        Inputs:
            - Dmat : ndarray, shape=(3N, 3N)
                The input square matrix, which is expected to be diagonalizable. In the context
                of phonon calculations, this is often the dynamical matrix (Hessian matrix).
        Outputs:
            - wsquare_indices : ndarray, shape=(3N,)
                The sorted eigenvalues of Dmat, corresponding to the squared frequencies (w2) in
                phonon calculations.
            - Wmat : ndarray, shape=(3N, 3N)
                A diagonal matrix constructed from the sorted eigenvalues. It represents the phonon
                frequency squared values along its diagonal.
            - Pmat : ndarray, shape=(3N, 3N)
                The matrix of sorted eigenvectors, which serves as the transformation matrix between
                the original coordinate system and the phonon coordinate system.
    """
    
    eigenvalues, eigenvectors = jnp.linalg.eigh(Dmat)

    indices_zero = jnp.where(jnp.abs(eigenvalues) < tolerance)[0]
    indices_nonzero = jnp.where(jnp.abs(eigenvalues) >= tolerance)[0]
    
    sorted_indices = jnp.concatenate([indices_zero, indices_nonzero])
    sorted_eigenvalues = jnp.take(eigenvalues, sorted_indices)
    sorted_eigenvectors = jnp.take(eigenvectors, sorted_indices, axis=1)

    wsquare_indices = sorted_eigenvalues
    Wmat = jnp.diag(sorted_eigenvalues)
    Pmat = sorted_eigenvectors
    
    num_zeros = len(indices_zero)
    if verbosity > 0:
        print("Number of zero frequencies: ", len(indices_zero))
    if num_zeros != 3:
        raise ValueError(f"Number of zero frequencies is not dim=3 !!!, it is {num_zeros}")  
    
    return wsquare_indices, Wmat, Pmat

####################################################################################################
## Solve the dynamical matrix using Fourier transformation (block-wise)
def get_Dmateigs_block(Dmat, 
                       shift_vectors, 
                       supercell_size, 
                       atom_per_unitcell
                       ):
    
    num_cells, dim = shift_vectors.shape
    kvector = 2 * jnp.pi * shift_vectors / supercell_size
    Fmat = jnp.exp(1j * jnp.einsum("id,jd->ij", kvector, shift_vectors))
    Fmat = jnp.kron(Fmat, jnp.eye(atom_per_unitcell*dim)) / jnp.sqrt(jnp.prod(supercell_size))

    Dmat = jnp.array(Dmat, dtype=jnp.complex128)
    Fmat = jnp.array(Fmat, dtype=jnp.complex128)

    Dblockdiag = Fmat.conj().T @ Dmat @ Fmat

    @partial(jax.jit)
    def solve_Dblock(block_index):
        
        start_pos = atom_per_unitcell * dim * block_index
        length = atom_per_unitcell * dim
        
        Dblock = jax.lax.dynamic_slice(Dblockdiag, (start_pos, start_pos), (length, length))
        Dblock_wsquare, Dblock_P = jnp.linalg.eigh(Dblock)
        Dblock_W = jnp.diag(Dblock_wsquare)

        return Dblock_wsquare, Dblock_W, Dblock_P

    vmap_solve_Dblock = jax.vmap(solve_Dblock, in_axes=(0))

    block_indices = jnp.arange(num_cells)
    Dblock_wsquares, Dblock_Ws, Dblock_Ps = vmap_solve_Dblock(block_indices)
    # Pmatrix = jax.scipy.linalg.block_diag(*Dblock_Ps)
    # Wmatrix = jax.scipy.linalg.block_diag(*Dblock_Ws)

    return Dblock_wsquares

####################################################################################################
## get phonon momentum of each block
def get_phonon_momentum(Dblock_wsquares, 
                        shift_vectors, 
                        wsquare_indices, 
                        tolerance=1e-4,
                        verbosity=1,
                        ):
    """
    Calculate the phonon momentum from wsquare indices and associated k-points.
    
    Inputs:
        - wsquare_indices (array): The input wsquare indices (frequencies squared).
        - Dblock_wsquares (array): The reference wsquare values (from dynamical matrix blocks).
        - shift_vectors (array): Associated k-point shift vectors.
        - verbosity (int): Level of verbosity for output (default is 1).

    Outputs:
        - w_indices_new (array): New w (frequencies) values.
        - wsquare_indices_new (array): New w^2 (frequency squared) values.
        - kpoint_indices_new (array): Indices corresponding to k-points.
    """
    
    # Define a tolerance value for zero-frequency detection
    if jnp.any(wsquare_indices < -1e-3) and verbosity > 0:
        print("Some negative frequencies detected, setting to positive.")
    
    # D block matrix sorting
    num_kpoints, num_bands = Dblock_wsquares.shape
    Dblock_sorted_indices = jnp.argsort(Dblock_wsquares.flatten())
    row_idx, col_idx = jnp.unravel_index(Dblock_sorted_indices, (num_kpoints, num_bands))
    Dblock_kpoints_list = jnp.array(row_idx)
    Dblock_wsquares_list = Dblock_wsquares[row_idx, col_idx]

    # D block sort zero frequencies to the top
    indices_zero = np.where(np.abs(Dblock_wsquares_list) < tolerance)[0]
    indices_nonzero = np.where(np.abs(Dblock_wsquares_list) >= tolerance)[0]
    sorted_indices = np.concatenate([indices_zero, indices_nonzero])
    Dblock_kpoints_list = jnp.take(Dblock_kpoints_list, sorted_indices)
    Dblock_wsquares_list = jnp.take(Dblock_wsquares_list, sorted_indices)

    # Get the new w, w^2 values, and k-point indices
    w_indices_new = jnp.sqrt(jnp.abs(Dblock_wsquares_list))
    wsquare_indices_new = Dblock_wsquares_list
    kpoint_indices_new  = Dblock_kpoints_list
    kpoint_vectors_new  = jnp.take(shift_vectors, Dblock_kpoints_list, axis=0)
    
    if verbosity > 0:
        print("eigs of Dblock & Dmat all close:", 
              jnp.allclose(Dblock_wsquares_list, wsquare_indices, atol=1e-4))
        
    if verbosity > 1:
        print("frequencies (w) are in the unit of (effective_mass @ Kelvin):")
        print("=====================================================")
        print(" idx  |     w      |      w^2      |  k-points")
        print("-----------------------------------------------------")
        for ii in range(len(wsquare_indices_new)):
            print(f"{ii:3d}   | {w_indices_new[ii]:10.6f} | "
                   f"{wsquare_indices_new[ii]:13.6f} | {kpoint_indices_new[ii]:3d}  "
                   + str(jnp.array(shift_vectors[kpoint_indices_new[ii]], dtype=jnp.int64)))
        print("=====================================================")
    
    return w_indices_new, wsquare_indices_new, kpoint_indices_new, kpoint_vectors_new
    

####################################################################################################
"""
This code performs a transformation of atomic coordinates within a supercell, utilizing relative
coordinates to convert phonon coordinates into atomic displacements, ultimately updating the 
atomic positions.

    R0    (shape: [N, 3]): Cartesian coordinates of atoms within the supercell.
    The equilibrium positions of the atoms are expressed in relative coordinates:
        R0base = R0[0:1, :]  (shape: [1, 3])   
            # Reference point, typically the origin or a chosen atom's equilibrium position.
        R0diff = R0[1:, :]   (shape: [N-1, 3])
            # Relative positions of the remaining atoms with respect to R0base.
        R0base, R0diff = jnp.split(R0, [1], axis=0)

    Q = U @ P    # Phonon coordinates (Q) are obtained by multiplying the displacement 
                    coordinates (U) with a transformation matrix (P).
    U = Q @ P.T  # Displacement coordinates (U) are derived by multiplying phonon 
                    coordinates (Q) with the transpose of the transformation matrix (P.T).
    R = R0 + U   # Atomic coordinates (R) are updated by adding the displacement 
                    coordinates (U) to the equilibrium positions (R0).
"""

## Transform from phonon/displacement coordinate to real space
def get_coordinate_transforms(num_atoms, 
                              dim = 3,
                              coordinate_type = "phon",
                              ):
    """
    This function generates coordinate transformation functions based on the type of input 
    coordinates (phonon or atomic).
    
    Inputs:
        - num_atoms (int): 
            The number of atoms in the system.
        - coordinate_type (str, default="phon"): 
            Specifies the type of input coordinates
                "phon" for phonon coordinates. 
                "atom" for displacement coordinates.

    Outputs:
        - trans_Q2R (function): 
            A function that transforms phonon (Q) or displacement (U) coordinates to real 
            space coordinates (R).
        - trans_R2Q (function): 
            A function that transforms real space coordinates (R) to phonon (Q) or 
            displacement (U) coordinates.
            
    Note:
        for simplify in this function: L = box_lengths
    """

    num_modes = (num_atoms - 1) * dim
    
    if coordinate_type == "phon":
        
        def trans_Q2R(Q, R0, L, Pmat):
            R0base, _ = jnp.split(R0, [1], axis=0)
            R0rela_flatten = (R0 - R0base).flatten()

            Q_flatten = jnp.concatenate((jnp.zeros((dim, )), Q.flatten()), axis=0)
            R_flatten = R0rela_flatten + Q_flatten @ Pmat.T
            R = R_flatten.reshape(num_atoms, dim)
            
            Rbase, _ = jnp.split(R, [1], axis=0)
            R = R - Rbase + R0base
            R = R - L * jnp.floor(R/L)
            return R
        
        def trans_R2Q(R, R0, L, Pmat):
            R0base, _ = jnp.split(R0, [1], axis=0)
            R0rela = (R0 - R0base)
            
            Rbase, _ = jnp.split(R, [1], axis=0)
            Rrela  = (R - Rbase)
            
            U = Rrela - R0rela
            U = U - L*jnp.rint(U/L)
            Q_flatten = U.flatten() @ Pmat
            Q = Q_flatten[dim:].reshape(num_modes, 1)
            return Q
    
    elif coordinate_type == "atom":
        raise ValueError("coordinate_type = 'atom' is discarded!")
    
    return trans_Q2R, trans_R2Q


####################################################################################################
def init_coordinate_transformations(coordinate_type, 
                                    Dmat, 
                                    shift_vectors = None,  
                                    supercell_size = None, 
                                    atom_per_unitcell = None, 
                                    num_atoms = None, 
                                    dim = 3, 
                                    cal_kpoint = True,
                                    verbosity = 0
                                    ):
    """
    Initialize coordinate transformations based on the specified type 
        (only phonon coordinates now).
    
    Inputs:
        - coordinate_type (str): Type of coordinate transformation, e.g., "phon" for phonons.
        - Dmat (array): Dynamical matrix or any relevant matrix used for the calculation.
        - shift_vectors (array): Shift vectors for supercell calculations.
        - supercell_size (int): Size of the supercell used in the simulation.
        - atom_per_unitcell (int): Number of atoms per unit cell.
        - num_atoms (int): Total number of atoms.
        - dim (int): Dimensionality of the system (default is 3).
        - cal_kpoint (bool): Whether to calculate k-points (default is True).
        - verbosity (int): Verbosity level for logging (default is 0).

    Outputs:
        - wsquare_indices (array): Squared frequencies of phonon modes.
        - w_indices (array): Frequencies of phonon modes.
        - Pmat (array): Eigenvectors from the diagonalization process.
        - num_modes (int): Number of phonon modes.

    Raises:
        ValueError: If an invalid coordinate type is passed or if there is a mismatch in the number 
        of modes.
    """
    
    if coordinate_type == "phon":
        if verbosity > 0: 
            print("Coordinate type: phonon")

        # Get eigenvalues and eigenvectors from Dmat
        wsquare_indices, Wmat, Pmat = get_Dmateigs(Dmat)
        num_modes = (num_atoms - 1) * dim
        
        if cal_kpoint: 
            # Calculate block matrices for phonon dispersion calculation
            Dblock_wsquares = get_Dmateigs_block(Dmat, 
                                                shift_vectors, 
                                                supercell_size, 
                                                atom_per_unitcell
                                                )

            # Obtain k-point indices and frequencies
            w_indices, wsquare_indices, kpoint_indices, kpoint_vectors \
                            = get_phonon_momentum(Dblock_wsquares, 
                                                shift_vectors,
                                                wsquare_indices,
                                                verbosity=verbosity
                                                )
            
            # Take the square root of the absolute values of the squared frequencies
            wkd_rawdata = (w_indices, wsquare_indices, kpoint_indices, kpoint_vectors, Dblock_wsquares)
            w_indices, wsquare_indices, kpoint_indices, kpoint_vectors \
                = w_indices[dim:], wsquare_indices[dim:], kpoint_indices[dim:], kpoint_vectors[dim:, :]
                        
            # Ensure the number of frequencies matches the expected number of modes
            if len(wsquare_indices) != num_modes or len(kpoint_indices) != num_modes:
                raise ValueError("Number of frequencies and k-points do not match !!!", 
                                len(wsquare_indices), len(kpoint_indices), num_modes)
                
        else:
            # Check for negative frequencies and adjust them
            if jnp.any(wsquare_indices < -1e-3) and verbosity > 0:
                print("Some negative frequencies detected, setting to positive.")
            w_indices = jnp.sqrt(jnp.abs(wsquare_indices))
            
            # Take the square root of the absolute values of the squared frequencies
            wkd_rawdata = (w_indices, wsquare_indices)
            w_indices, wsquare_indices = w_indices[dim:], wsquare_indices[dim:]
            
            # Ensure the number of frequencies matches the expected number of modes
            if len(wsquare_indices) != num_modes:
                raise ValueError("Number of frequencies do not match !!!", 
                                len(wsquare_indices), num_modes)


    elif coordinate_type == "atom":
        raise ValueError("coordinate_type = 'atom' is deprecated!")
    else:
        raise ValueError("Invalid coordinate_type!")

    return wsquare_indices, w_indices, Pmat, wkd_rawdata, num_modes

####################################################################################################




####################################################################################################
### This is the end of this file
































# ####################################################################################################
# ## get phonon momentum of each block
# def get_phonon_momentum(Dblock_wsquares, 
#                         shift_vectors, 
#                         wsquare_indices, 
#                         verbosity=1,
#                         wfac=1e6, 
#                         wtol=1e-6,
#                         ):
#     """
#     Calculate the phonon momentum from wsquare indices and associated k-points.
    
#     Inputs:
#         - wsquare_indices (array): The input wsquare indices (frequencies squared).
#         - Dblock_wsquares (array): The reference wsquare values (from dynamical matrix blocks).
#         - shift_vectors (array): Associated k-point shift vectors.
#         - verbosity (int): Level of verbosity for output (default is 1).

#     Outputs:
#         - w_indices_new (array): New w (frequencies) values.
#         - wsquare_indices_new (array): New w^2 (frequency squared) values.
#         - kpoint_indices_new (array): Indices corresponding to k-points.
#     """
    
#     # Scale wsquare indices for better precision and round them
#     wsquare_indices = jnp.round(wsquare_indices / wfac) * wfac
#     wsquare_unique, wsquare_counts = jnp.unique(wsquare_indices, return_counts=True)
    
#     # Define a tolerance value for zero-frequency detection
#     indices_zero = jnp.where(jnp.abs(wsquare_unique) < 1e-4)[0]
#     indices_nonzero = jnp.where(jnp.abs(wsquare_unique) >= 1e-4)[0]
    
#     # Sort zero-frequency indices first, followed by non-zero frequencies
#     sorted_indices = jnp.concatenate([indices_zero, indices_nonzero])
#     wsquare_unique = wsquare_unique[sorted_indices]
    
#     # Check for negative frequencies and log a warning if found
#     if jnp.any(wsquare_indices < -1e-3) and verbosity > 0:
#         print("Some negative frequencies detected, setting to positive.")
    
#     w_indices_new = []
#     wsquare_indices_new = []
#     kpoint_indices_new = []
    
#     if verbosity > 0:
#         print("frequencies (w) are in the unit of (effective_mass @ Kelvin):")
#         print("===================================================================================")
#         print(" idx  |     w      |      w^2      | deg |  k-points")
#         print("-----------------------------------------------------------------------------------")
    
#     # Iterate over each unique wsquare index to compute frequencies and match k-points
#     for ii in range(len(wsquare_unique)):
#         wsquare_temp = wsquare_unique[ii]
#         w_temp = jnp.sqrt(jnp.abs(wsquare_temp))
        
#         condition = jnp.abs(wsquare_unique[ii] - Dblock_wsquares) < wtol
#         indices = jnp.where(condition)[0]

#         deg = len(indices)
#         if verbosity > 0:
#             print(f"{ii:3d}   | {w_temp:10.6f} | {wsquare_temp:13.6f} | {deg:3d} |" +
#                   "  ".join(f"{x:3d}" for x in indices))
            
#         for jj in range(len(indices)):
#             w_indices_new.append(w_temp)
#             wsquare_indices_new.append(wsquare_temp)
#             kpoint_indices_new.append(indices[jj])
            
#     if verbosity > 0:
#         print("===================================================================================")
     
#     w_indices_new = jnp.array(w_indices_new)
#     wsquare_indices_new = jnp.array(wsquare_indices_new)
#     kpoint_indices_new = jnp.array(kpoint_indices_new)
    
#     return w_indices_new, wsquare_indices_new, kpoint_indices_new