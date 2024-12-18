import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

####################################################################################################
"""
    This script calculates the pressure of a system based on the approach outlined in 
    Phys. Rev. B 32, 3780 (1985). The implementation is adapted from the JAX-MD library, 
    specifically from the quantity module:
        https://jax-md.readthedocs.io/en/main/_modules/jax_md/quantity.html#
        
    Unit conversion:
        convert K/A^3 to GPa using the factor: 8.617333262145e-5 * 1.602176634e2
        1K/A^3 = 0.013806489999999715 GPa
"""

def pressurefn_novmap(energyfn, coord, box_lengths, kinetic_energy):
    """
    Computes the pressure of a system using the formula:
        P = (2 * K - dU/de) / (d * Vol)
    Inputs: 
        - energyfn : function
            A function that computes the energy with atomic coordinates and box lengths.
            The energy is expected to be in units of Kelvin (K).
        - coord : ndarray, shape=(N, 3)
            The Cartesian coordinates of the atoms in the system, where N is the number of atoms.
        - box_lengths : ndarray, shape=(3,)
            The lengths of the simulation box in each dimension, given in angstroms (A).
        - kinetic_energy : float
            The kinetic energy of the system in Kelvin (K).
    Outputs: 
        - pressure: float
            The computed pressure of the system in units of K/A^3.
    """
    
    num_atoms, dim = coord.shape
    
    def U_fn(eps): 
        # eps: A small strain parameter applied uniformly to the system.
        coord_eps = coord * (1.0 + eps)
        box_lengths_eps = box_lengths * (1.0 + eps)
        return energyfn(coord_eps, box_lengths_eps)
    
    # Gradient of the energy with respect to the strain.
    gradU_fn = jax.grad(U_fn)
    box_volume = jnp.prod(box_lengths)
    
    # Calculate the pressure using the given formula.
    pressure = 1 / (dim * box_volume) * (2 * kinetic_energy - gradU_fn(0.0))
    return pressure

pressurefn = jax.vmap(pressurefn_novmap, in_axes=(None, 0, None, 0), out_axes=(0))

####################################################################################################

def stressfn_novmap(energyfn, coord, box_lengths, kinetic_energy, forloop=True):
    """
    Computes the stress tensor of a system. 
        This implementation assumes an orthogonal simulation box and 
        approximates kinetic terms <psi|pxpy|psi> using Ek/3.
    Inputs: 
        - energyfn : function
            A function that computes the energy with atomic coordinates and box lengths.
            The energy is expected to be in units of Kelvin (K).
        - coord : ndarray, shape=(N, 3)
            The Cartesian coordinates of the atoms in the system, where N is the number of atoms.
        - box_lengths : ndarray, shape=(3,)
            The lengths of the simulation box in each dimension, given in angstroms (A).
        - kinetic_energy : float
            The kinetic energy of the system in Kelvin (K).
    Outputs: 
        - stress : ndarray, shape=(3,)
            The computed stress tensor of the system in units of K/A^3.
    """

    num_atoms, dim = coord.shape
    box_volume = jnp.prod(box_lengths)
    # Zero and identity vectors for the strain perturbation in each dimension.
    I = jnp.ones((dim, ), dtype=jnp.float64)
    zero = jnp.zeros((dim, ), dtype=jnp.float64)
    
    if forloop:
        def U_fn(eps, index):
            eps_vector = zero.at[index].set(eps)
            coord_eps = coord * (I + eps_vector)
            box_lengths_eps = box_lengths * (I + eps_vector)
            return energyfn(coord_eps, box_lengths_eps)
        
        def compute_grad(index, dUde):
            # grad_value = jax.grad(lambda eps: U_fn(eps, index))(0.0)
            grad_value = jax.jacfwd(lambda eps: U_fn(eps, index))(0.0)
            return dUde.at[index].set(grad_value)

        dUde = jax.lax.fori_loop(0, dim, compute_grad, zero)
        stress = 1 / box_volume * (2 * kinetic_energy / dim - dUde)
 
    else:
        def U_fn(eps): 
            coord_eps = coord * (I + eps)
            box_lengths_eps = box_lengths * (I + eps)
            return energyfn(coord_eps, box_lengths_eps)
        # Gradient of the energy with respect to the strain
        gradU_fn = jax.grad(U_fn)
        stress = 1 / box_volume * (2 * kinetic_energy / dim - gradU_fn(zero))
        
    return stress

stressfn = jax.vmap(stressfn_novmap, in_axes=(None, 0, None, 0, None), out_axes=(0))

####################################################################################################
if __name__ == "__main__":
    from crystal_lithium import create_supercell
    from dpjax_lithium import make_dp_model
    import time
    
    key = jax.random.key(42)
    
    batch = 32
    dim = 3
    lattice_type = "fcc"
    volume_per_atom = 18.0
    supercell_size = [2, 2, 2]
    pkl_name = "src/dpmodel/fznp_dp0.pkl"
    
    R0, box_lengths, num_atoms = create_supercell(lattice_type, volume_per_atom, supercell_size)
    dp_energyfn, dp_forcefn = make_dp_model(pkl_name, num_atoms, box_lengths_init=box_lengths, unit="K")
    
    U = 0.5 * jax.random.uniform(key, (batch, num_atoms, dim))
    R = R0 + U
    L = box_lengths
    coord = R - L * jnp.floor(R/L)
    kinetic_energy = 100.0 * jnp.ones((batch, ))
    print("coordinate:", coord.shape)
    
    pressure = pressurefn(dp_energyfn, coord, box_lengths, kinetic_energy)
    print("pressure:", pressure.shape, pressure.mean())

    def calculate_stress():
        t1 = time.time()
        forloop = True
        stress = stressfn(dp_energyfn, coord, box_lengths, kinetic_energy, forloop)
        t2 = time.time()
        print("foriloop version dt:", t2 - t1)
        print("stress:", stress.shape, stress.mean(axis=0))
        print("trace(stress):", stress.mean())

        t1 = time.time()
        forloop = False
        stress = stressfn(dp_energyfn, coord, box_lengths, kinetic_energy, forloop)
        t2 = time.time()
        print("foriloop version dt:", t2 - t1)
        print("stress:", stress.shape, stress.mean(axis=0))
        print("trace(stress):", stress.mean())
        
    ### first time
    print("\ntest first time")
    calculate_stress()
    ### again
    print("\ntest again")
    calculate_stress()
    print("\ntest again")
    calculate_stress() 
    
    
"""
cd /home/zhangqi/MLCodes/lithium/
cd /mnt/t02home/MLCodes/lithium/
python3 src/quantity.py
"""














####################################################################################################
        # def Ux_fn(epsx): 
        #     eps = jnp.array([epsx, 0.0, 0.0])
        #     coord_eps = coord * (I + eps)
        #     box_lengths_eps = box_lengths * (I + eps)
        #     return energyfn(coord_eps, box_lengths_eps)
        
        # def Uy_fn(epsy): 
        #     eps = jnp.array([0.0, epsy, 0.0])
        #     coord_eps = coord * (I + eps)
        #     box_lengths_eps = box_lengths * (I + eps)
        #     return energyfn(coord_eps, box_lengths_eps)

        # def Uz_fn(epsz): 
        #     eps = jnp.array([0.0, 0.0, epsz])
        #     coord_eps = coord * (I + eps)
        #     box_lengths_eps = box_lengths * (I + eps)
        #     return energyfn(coord_eps, box_lengths_eps)
        
        # box_volume = jnp.prod(box_lengths)
        # gradUx = jax.grad(Ux_fn)(0.0)
        # gradUy = jax.grad(Uy_fn)(0.0)
        # gradUz = jax.grad(Uz_fn)(0.0)
        
        # dUde = jnp.array([gradUx, gradUy, gradUz])
        # stress = 1 / box_volume * (2 * kinetic_energy / dim - dUde)
 
 
 
 
        #  def U_fn(eps, index): 
        #     eps_vector = jnp.zeros((dim,), dtype=jnp.float64).at[index].set(eps)
        #     coord_eps = coord * (I + eps_vector)
        #     box_lengths_eps = box_lengths * (I + eps_vector)
        #     return energyfn(coord_eps, box_lengths_eps)
        
        # box_volume = jnp.prod(box_lengths)
        # dUde = jnp.array([jax.grad(lambda eps: U_fn(eps, i))(0.0) for i in range(dim)])
        # stress = 1 / box_volume * (2 * kinetic_energy / dim - dUde)
        
        
        
        # def U_fn(eps, index):
        #     eps_vector = zero.at[index].set(eps)
        #     coord_eps = coord * (I + eps_vector)
        #     box_lengths_eps = box_lengths * (I + eps_vector)
        #     return energyfn(coord_eps, box_lengths_eps)
        
        # def compute_grad(i, dUde):
        #     grad_value = jax.grad(lambda eps: U_fn(eps, i))(0.0)
        #     return dUde.at[i].set(grad_value)

        # dUde = jax.lax.fori_loop(0, dim, compute_grad, zero)
        # stress = 1 / box_volume * (2 * kinetic_energy / dim - dUde)