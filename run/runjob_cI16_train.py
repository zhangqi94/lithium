import os
import sys
import subprocess
import runtools
        
####################################################################################################
cpus = "12"
nodelist = ""

# nodelist = "node005"
# partition = "a800"
# gpus = "4"

# nodelist = "node003"
# nodelist = "node004"

# partition = "v100"
# gpus = "8"
partition = "a800"
gpus = "4"

####################################################################################################
list_volume = [7.2, 7.1, 6.9, 6.8, 6.7, 6.6, 6.4, 6.3]

for ii in range(len(list_volume)):
    #========== lattice ==========
    target_pressure = "-1"
    supercell_length = "-1 -1 -1"

    ensemble = "nvt"
    volume_per_atom = str(list_volume[ii])

    isotope, supercell_size = "Li", "3 3 3"
    lattice_type, coordinate_type = "cI16", "phon"
    batch, acc_steps = "256", "1"
    
    relax_lr = "0.01"
    relax_min_lr = "0.01"
    relax_steps = "6000"
    relax_decay = "0.90"
    relax_begin = "6000"
    epoch = "15000"

    #========== batch ==========
    # batch, acc_steps = "1024", "1"

    seed = "16" ###key
    dptype = "dp0"
    indices_group, num_levels = "1", "20"

    #========== van ==========
    # temperature = "300"
    # temperature = "200"
    temperature = "100"

    # van_type = "van_tfhaiku"
    van_layers, van_size, van_heads, van_hidden = "4", "16", "4", "64"
    # van_layers, van_size, van_heads, van_hidden = "4", "32", "4", "128"
    # lr_class = "1e-3"
    # min_lr_class = "1e-6"

    van_type = "van_probtab"
    lr_class = "1e-2"
    min_lr_class = "1e-4"

    #========== flow ==========
    flow_depth, mlp_width, mlp_depth = "6", "128", "2"
    flow_type = "flw_rnvpflax"
    flow_st = "st"
    lr_quant = "1e-4"
    min_lr_quant = "5e-6"

    # flow_type = "flw_identity"
    # flow_st = "o"
    # lr_quant = "1e-3"
    # min_lr_quant = "1e-3"

    decay_rate = "0.97"
    decay_steps = "100"
    decay_begin = "1000"

    #========== structure relaxation ==========
    # for oC88
    # relax_lr = "0.0040"
    # relax_min_lr = "0.0010"
    # relax_steps = "1500"
    # relax_decay = "0.90"
    # relax_begin = "1500"

    relax_therm = "5"
    num_recent_vals = "50"
    if lattice_type in {"oC88", "oC40"}:
        cal_stress = "2"
    elif lattice_type in {"bcc", "fcc", "cI16"}:
        cal_stress = "1"

    #========== mcmc ==========
    mc_therm, mc_steps, mc_stddev = "10", "3000", "0.03"

    ckpt_epochs = "1000"
    # epoch = "30000"

    # hessian_type = "jax"
    hessian_type = "for1"
    load_ckpt = "None"

    folder = "/data/zhangqidata/lithium/"

    ####################################################################################################
    #========== job name ==========
    if van_type == "van_tfhaiku":
        van_str = f"_van[{van_layers}_{van_size}_{van_heads}_{van_hidden}]"
    elif van_type == "van_probtab":
        van_str = "_vanpt"
    else:
        raise ValueError("Invalid van_type")

    if flow_type == "flw_rnvpflax":
        flow_str = f"_flw[{flow_depth}_{mlp_width}_{mlp_depth}_{flow_st}]"
    elif flow_type == "flw_identity":
        flow_str = f"_flw[id_{flow_st}]"
    else:
        raise ValueError("Invalid flow_type")

    if relax_steps != '0' and ensemble == "npt":
        relax_str = f"_rlx[{relax_lr}_{relax_min_lr}_{relax_decay}_{relax_steps}_{relax_therm}_{num_recent_vals}]"
    else:
        relax_str = ""

    ####################################################################################################
    job_name = (
        f"job_{isotope}_{dptype}_{lattice_type}_t{temperature}"
        f"{'_npt_' + target_pressure if ensemble == 'npt' else ''}"
        f"{'_nvt_' + volume_per_atom if ensemble == 'nvt' else ''}"
        f"_n{supercell_size.replace(' ', '')}"
        f"_lev[{num_levels}_{indices_group}]"
        f"{van_str}{flow_str}"
        f"_mc[{mc_steps}_{mc_stddev}]"
        f"_lr[{lr_class}_{lr_quant}_{min_lr_class}_{min_lr_quant}_{decay_rate}_{decay_steps}]"
        f"{relax_str}"
        f"_bth[{batch}_{acc_steps}]"
        f"_k{seed}"
    )

    command = f"""python3 main.py  --executor "train"  \
            --seed {seed}  \
            --folder "{folder}"  \
            --load_ckpt "{load_ckpt}"  \
            --ckpt_epochs {ckpt_epochs}  \
            --isotope "{isotope}"  \
            --dpfile "{dptype}"  \
            --supercell_size {supercell_size}  \
            --lattice_type "{lattice_type}"  \
            --coordinate_type "{coordinate_type}"  \
            --temperature {temperature}  \
            --indices_group {indices_group}  \
            --num_levels {num_levels}  \
            --lr_class {lr_class}  \
            --lr_quant {lr_quant}  \
            --min_lr_class {min_lr_class}  \
            --min_lr_quant {min_lr_quant}  \
            --decay_rate {decay_rate}  \
            --decay_steps {decay_steps}  \
            --decay_begin {decay_begin}  \
            --hutchinson  \
            --hessian_type "{hessian_type}"  \
            --mc_therm {mc_therm}  \
            --mc_steps {mc_steps}  \
            --mc_stddev {mc_stddev}  \
            --batch {batch}  \
            --acc_steps {acc_steps}  \
            --epoch {epoch}  """

    if flow_type == "flw_rnvpflax":
        command = command + \
            f"""--flow_type "{flow_type}"  \
            --flow_depth {flow_depth}  \
            --mlp_width {mlp_width}  \
            --mlp_depth {mlp_depth}  \
            --flow_st  {flow_st}  """
    elif flow_type == "flw_identity":
        command = command + \
            f"""--flow_type "{flow_type}"  \
            --flow_st  {flow_st}  """

    if van_type == "van_tfhaiku":
        command = command + \
            f"""--van_type "{van_type}"  \
            --van_layers {van_layers}  \
            --van_size {van_size}  \
            --van_heads {van_heads}  \
            --van_hidden {van_hidden}  """
    elif van_type == "van_probtab":
        command = command + \
            f"""--van_type "{van_type}"  """

    if ensemble == "npt":
        command = command + \
            f"""--ensemble "{ensemble}"  \
            --target_pressure {target_pressure}  \
            --volume_per_atom {volume_per_atom}  \
            --supercell_length {supercell_length}  \
            --relax_begin {relax_begin}  \
            --relax_steps {relax_steps}  \
            --relax_therm {relax_therm}  \
            --relax_lr {relax_lr}  \
            --relax_min_lr {relax_min_lr}  \
            --relax_decay {relax_decay}  \
            --num_recent_vals {num_recent_vals}  \
            --cal_stress {cal_stress} """
    elif ensemble == "nvt":
        command = command + \
            f""" --ensemble "{ensemble}"  \
            --volume_per_atom {volume_per_atom} """


    gpustr = f"""####SBATCH --nodelist={nodelist}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
####SBATCH --cores-per-socket={int(cpus)//2}
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
"""

    ####################################################################################################

    # slurm_script = runtools.generate_slurm_script_jax0431(gpustr, command)
    # slurm_script = runtools.generate_slurm_script_jax0429(gpustr, command)
    slurm_script = runtools.generate_slurm_script_singularity(gpustr, command)
    file_name = job_name + ".sh"

    print("job_name:", job_name)
    # print("command:", command)
    # print("slurm_script:", slurm_script)
    # print("file_name:", file_name)  

    runtools.write_slurm_script_to_file(slurm_script, file_name)
    runtools.submit_slurm_script(file_name)
