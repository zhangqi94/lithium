import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
# from tools import load_structures

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

    Q = U @ P    # Phonon coordinates (Q) are obtained by multiplying the displacement 
                    coordinates (U) with a transformation matrix (P).
    U = Q @ P.T  # Displacement coordinates (U) are derived by multiplying phonon 
                    coordinates (Q) with the transpose of the transformation matrix (P.T).
    R = R0 + U   # Atomic coordinates (R) are updated by adding the displacement 
                    coordinates (U) to the equilibrium positions (R0).
"""

def get_phys_const(isotope="Li"):
    """
    Calculate physical constants needed for calculations involving helium.

    Outputs:
        - h2_over_2m (float): The value of (hbar^2)/(2*m*kB) in Kelvin/Angstrom^2, 
                            where hbar is the reduced Planck's constant,
                            m is the mass of atom, and
                            kB is the Boltzmann constant.
        - effective_mass (float): The effective mass calculated as the reciprocal of 
                                twice the value of (hbar^2)/(2*m*kB).
    """
    # Reduced Planck's constant (in Joule-second)
    hbar = 6.62607015e-34 / (2 * jnp.pi)
    # Boltzmann constant (in Joule/Kelvin)
    kB = 1.380649e-23
    # Mass of a lithium atom (kg)
    if isotope == "Li":
        m = 6.941 * 1.66053906660e-27
    if isotope == "Li6":
        m = 6.015122795 * 1.66053906660e-27
    if isotope == "Li7":
        m = 7.016004550 * 1.66053906660e-27
        
    # (hbar^2) / (2 * mass * Boltzmann constant) in Kelvin/Angstrom^2
    h2_over_2m = (hbar ** 2) / (2 * m * kB) * (10 ** 20)  
    # Effective mass in atomic units
    effective_mass = 1 / (2 * h2_over_2m)
    
    return h2_over_2m, effective_mass

####################################################################################################
def load_structures(structure_file):
    with open(structure_file, 'r') as file:
        lines = file.readlines()
    
    ## get cell length
    cell_index = lines.index('1.0\n')
    cell = []
    for i in range(cell_index+1, cell_index+4):
        cell.append(list(map(float, lines[i].split())))
    cell = np.array(cell, dtype=np.float64)
    
    ## get fractional coordinates
    direct_index = lines.index('Direct\n')
    coord = []
    for i in range(direct_index + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            break
        if line:
            coord.append(list(map(float, line.split())))
    coord = np.array(coord, dtype=np.float64)

    return cell, coord

####################################################################################################
lattice_types_dict = {
    "bcc": {"shift": [0.25, 0.25, 0.25], "file": "bcc.vasp"},
    "fcc": {"shift": [0.25, 0.25, 0.25], "file": "fcc.vasp"},
    "cI16": {"shift": [0.02, 0.02, 0.02], "file": "cI16.vasp"},
    "oC88": {"shift": [0, 0, 0], "file": "oC88.vasp"},
    "oC40": {"shift": [0, 0, 0], "file": "oC40.vasp"},
    "oP48": {"shift": [0, 0, 0], "file": "oP48.vasp"},
    "oP192": {"shift": [0, 0, 0], "file": "oP192.vasp"},
    "tI20": {"shift": [0, 0, 0], "file": "tI20.vasp"},
    "oC24_No41v1": {"shift": [0, 0, 0], "file": "oC24_No41v1.vasp"},
    "oC24_No41v2": {"shift": [0, 0, 0], "file": "oC24_No41v2.vasp"},
    "oC40_No64": {"shift": [0, 0, 0], "file": "oC40_No64.vasp"},
    "oP24_No61": {"shift": [0, 0, 0], "file": "oP24_No61.vasp"},
    "cI16_x0.055": {"shift": [0.02, 0.02, 0.02], "file": "cI16_x0.055.vasp"},
    "cI16_x0.195": {"shift": [0.02, 0.02, 0.02], "file": "cI16_x0.195.vasp"},
    "cI16_p70": {"shift": [0.02, 0.02, 0.02], "file": "cI16_p70.vasp"},
    "oC88_p70": {"shift": [0, 0, 0], "file": "oC88_p70.vasp"},
    "oC40_p70": {"shift": [0, 0, 0], "file": "oC40_p70.vasp"},
}

def create_supercell(lattice_type = "fcc",
                     volume_per_atom = 15, 
                     supercell_size = [2, 2, 2],
                     supercell_length = [-1, -1, -1],
                     structure_path = "src/structures",
                     ):
    """
    Define unit cell dimensions and initial atomic positions.
    Note: Only orthogonal (rectangular) lattice structures are supported now!
    R0base, R0diff = jnp.split(R0, [1], axis=0)
    """
    
    supercell_size = np.array(supercell_size, dtype=np.int64)
    supercell_length = np.array(supercell_length, dtype=np.float64)
    
    # ========== Define supported lattice types ==========    
    if lattice_type not in lattice_types_dict:
        raise ValueError(f"Unsupported lattice type: {lattice_type}")
    
    # ========== Load structure data and apply shifts ==========
    lattice_info = lattice_types_dict[lattice_type]
    R0_shift = np.array(lattice_info["shift"], dtype=np.float64)
    L0_unitcell, R0_unitcell = load_structures(f"{structure_path}/{lattice_info['file']}")
    # L0_unitcell, R0_unitcell = load_structures(f"src/structures/{lattice_info['file']}")

    # ========== Handle supercell length logic ==========
    if np.all(supercell_length > 0):  # Use provided supercell length
        L0_unitcell = supercell_length / supercell_size
    else:  # Use unit cell's dimensions if no valid supercell length provided
        L0_unitcell = np.diag(L0_unitcell)
    
    # Calculate the number of atoms and update volume per atom if necessary
    supercell_size = np.array(supercell_size, dtype=np.int64)
    atom_per_unitcell = len(R0_unitcell)
    num_atoms = np.prod(supercell_size) * atom_per_unitcell

    if np.all(supercell_length > 0):
        volume_per_atom = np.prod(supercell_length) / num_atoms
    
    # R0_unitcell = L0_unitcell * (R0_unitcell + R0_shift)
    R0_unitcell = L0_unitcell * R0_unitcell
    
    # ========== Initialize supercell ==========
    # Generate shifts for the supercell
    supercell_size_x, supercell_size_y, supercell_size_z = supercell_size 
    index_x, index_y, index_z = np.mgrid[0:supercell_size_x, 0:supercell_size_y, 0:supercell_size_z]
    shift_vectors = np.vstack((index_x.flatten(), index_y.flatten(), index_z.flatten())).T
    shift_vectors = np.array(shift_vectors, dtype=np.float64)
    supercell_shifts = shift_vectors[:, None, :] * L0_unitcell

    # Broadcast to generate all positions & Flatten to list of positions
    R0_supercell_nov = (np.array(R0_unitcell + supercell_shifts)).reshape(-1, 3) 
    # Calculate total length of the super cell and number of atoms
    L0_supercell_nov = L0_unitcell * supercell_size 

    # Calculate volumes (in the unit of A^3) and get zoom in factor
    volume_unitcell_nov = np.prod(L0_unitcell)
    volume_unitcell = atom_per_unitcell * volume_per_atom
    zoom_in_factor = (volume_unitcell / volume_unitcell_nov) ** (1/3)

    # Scale L and R0 by lattice constant
    L0_supercell = L0_supercell_nov * zoom_in_factor
    R0_supercell = R0_supercell_nov * zoom_in_factor
    
    # change variable names
    box_lengths = jnp.array(L0_supercell, dtype=jnp.float64) 
    R0          = jnp.array(R0_supercell, dtype=jnp.float64)
    R0 = R0 - box_lengths * jnp.floor(R0 / box_lengths)

    # ========== Adjust box lengths for specific lattice types ==========
    box_lengths_old = box_lengths.copy()
    if lattice_type in {"cI16", "fcc", "bcc"}:
        average_value = box_lengths.mean()
        box_lengths = jnp.full_like(box_lengths, average_value)        
        
    elif lattice_type in {"tI20"}:
        average_of_first_two = box_lengths[:2].mean()
        box_lengths = box_lengths.at[:2].set(average_of_first_two) 
        
    elif lattice_type in {"oC88", "oC40", "oP48", "oP192"}:
        pass

    # Rescale R0 based on the updated box lengths
    R0 = R0 / box_lengths_old * box_lengths

    return R0, box_lengths, num_atoms, shift_vectors, atom_per_unitcell

####################################################################################

def estimated_volume_from_pressure(pressure):
    # Calculate the volume using the pressure and coefficients.
    a, b, c0, c1, c2, c3 = [ 1.42968085e+02,  1.10012398e+01,  7.16513098e+00, 
                            -2.68480269e-02,  8.52741727e-06,  4.45521304e-07]
    volume = c0 + a / (pressure + b) + c1 * pressure + c2 * pressure**2 + c3 * pressure**3
    return volume

####################################################################################################

def get_shift_vectors(lattice_type, 
                      supercell_size, 
                      structure_path = "/mnt/t02home/MLCodes/lithium/src/structures",
                      ):

    R0, box_lengths, num_atoms, shift_vectors, atom_per_unitcell = create_supercell(lattice_type, 
                                                                supercell_size = supercell_size, 
                                                                structure_path = structure_path
                                                                )
        
    return shift_vectors, atom_per_unitcell

####################################################################################################





### This is the end of this code.



