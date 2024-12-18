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

current_dir = os.getcwd()
print("current_dir:", current_dir)
sys.path.append(current_dir)
sys.path.append(current_dir + "/src")
sys.path.append(current_dir + "/executor")

####################################################################################################
jnp.set_printoptions(precision=6)

# import subprocess
# print(subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True).stdout, flush=True)
print("jax.version:", jax.__version__, flush=True)
print("optax.version:", optax.__version__, flush=True)
print("haiku.version:", haiku.__version__, flush=True)
print("flax.version:", flax.__version__, flush=True)
print("""
██╗     ██╗████████╗██╗  ██╗██╗██╗   ██╗███╗   ███╗
██║     ██║╚══██╔══╝██║  ██║██║██║   ██║████╗ ████║
██║     ██║   ██║   ███████║██║██║   ██║██╔████╔██║
██║     ██║   ██║   ██╔══██║██║██║   ██║██║╚██╔╝██║
███████╗██║   ██║   ██║  ██║██║╚██████╔╝██║ ╚═╝ ██║
╚══════╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝     ╚═╝                                      
""")

####################################################################################################
import argparse
parser = argparse.ArgumentParser(description= "finite-temperature simulation for solid lithium")

# specify the execution mode
"""
    1. train: training the flow (wave function) and van (probability) model
    2. spehar: calculate the harmonic spectra
    3. speflw: calculate the spectra with flow model
    4. speflwvan: calculate the spectra with flow and van model
"""
parser.add_argument("--executor", type=str, default="train", help="train, spehar, speflw")

# folder to save data
parser.add_argument("--folder", default="/data/zhangqidata/check", 
                    help="the folder to save data & checkpoints"
                    )
parser.add_argument("--load_ckpt", type=str, default=None, 
                    help="load checkpoint"
                    )
parser.add_argument("--ckpt_epochs", type=int, default=20, 
                    help="save checkpoint per ckpt_epochs"
                    )

# physical parameters of the crystal
parser.add_argument("--isotope", type=str, default="Li", 
                    help="isotope: Li, Li6, Li7"
                    )
parser.add_argument("--dpfile", type=str, default="dp0", 
                    help="dpmodel file: dp0 (Li-DP-Hyb2 model), dp4 (Li-DP-Hyb3 model)"
                    )

parser.add_argument("--ensemble", type=str, default="nvt", 
                    help="nvt or npt ensemble"
                    )
parser.add_argument("--target_pressure", type=float, default=3.0, 
                    help="target pressure (GPa) (only for npt ensemble!!!)"
                    )
parser.add_argument("--volume_per_atom", type=float, default=-1.0, 
                    help="volume per atom (A^3) (for npt or nvt)"
                    )
parser.add_argument("--supercell_length", type=float, nargs=3, default=[-1, -1, -1],
                    help="initial supercell length (only for npt ensemble!!!)"
                    )

parser.add_argument("--supercell_size", type=int, nargs=3, default=[3, 3, 3], 
                    help="supercell size: number of unit cells in each direction"
                    )
parser.add_argument("--lattice_type", type=str, default="fcc", 
                    help="support lattice type: bcc, fcc, cI16, oC88, oC40")
parser.add_argument("--coordinate_type", type=str, default="phon", 
                    help="coordinate: atom or phon (only phonon coordinates is supported now)"
                    )
parser.add_argument("--dim", type=int, default=3, 
                    help="dimension (only dim = 3 is supported)"
                    )

# parameters of the autoregressive model
# van_type: van_tfhaiku, van_probtab
parser.add_argument("--temperature", type=float, default=100.0, 
                    help="temperature (Kelvin)"
                    )
parser.add_argument("--num_levels", type=int, default=10, 
                    help="number of levels"
                    )
parser.add_argument("--indices_group", type=int, default=3, 
                    help="number of indices in a group"
                    )
parser.add_argument("--van_type", type=str, default="van_tfhaiku", 
                    help="probabilistic model: \
                        van_tfhaiku (variational autoregressive model), \
                        van_probtab (product spectrum ansatz)"
                    )
parser.add_argument("--van_layers", type=int, default=2, 
                    help="number of layers"
                    )
parser.add_argument("--van_size", type=int, default=16, 
                    help="size of the model"
                    )
parser.add_argument("--van_heads", type=int, default=4, 
                    help="number of heads"
                    )
parser.add_argument("--van_hidden", type=int, default=32, 
                    help="number of hidden units"
                    )

# parameters of the flow model
# flow_type: flw_rnvpflax, flw_identity
parser.add_argument("--flow_type", type=str, default="flw_rnvpflax", 
                    help="written in flax"
                    )
parser.add_argument("--flow_depth", type=int, default=8, 
                    help="depth of the flow model"
                    )
parser.add_argument("--mlp_width", type=int, default=64, 
                    help="width of the mlp layers"
                    )
parser.add_argument("--mlp_depth", type=int, default=2, 
                    help="depth of the mlp layers"
                    )
parser.add_argument("--flow_st", type=str, default="st", 
                    help="add scaling and shift to flow model: st, o"
                    )

# parameters of the optimizer
parser.add_argument("--lr_class", type=float, default=1e-2, 
                    help="learning rate classical model"
                    )
parser.add_argument("--lr_quant", type=float, default=1e-2, 
                    help="learning rate quantum model"
                    )
parser.add_argument("--min_lr_class", type=float, default=1e-6, 
                    help="minimum learning rate classical model"
                    )
parser.add_argument("--min_lr_quant", type=float, default=1e-6, 
                    help="minimum learning rate quantum model"
                    )
parser.add_argument("--decay_rate", type=float, default=0.97, 
                    help="decay rate of the learning rate"
                    )
parser.add_argument("--decay_steps", type=int, default=100, 
                    help="decay steps of the learning rate"
                    )
parser.add_argument("--decay_begin", type=int, default=500, 
                    help="epochs to start decay"
                    )

parser.add_argument("--hutchinson", action='store_true', 
                    help="Hutchinson's trick"
                    )
parser.add_argument("--clip_factor", type=float, default=5.0, 
                    help="clip factor"
                    )
parser.add_argument("--cal_stress", type=int, default=0, 
                    help="calculate stress mode: \
                        0 (no stress), \
                        1 (all directions are same), \
                        2 (three directions are different)"
                    )
parser.add_argument("--hessian_type", type=str, default="for1", 
                    help="method to calculate hessian: jax, for1 (use foriloop per column), for2"
                    )

# optimizer only for npt ensemble (geometry optimization, structural relaxation)!!!!!!
parser.add_argument("--relax_begin", type=int, default=1000,
                    help="disable cell length updates after the specified number of epochs"
                    )
parser.add_argument("--relax_steps", type=int, default=0, 
                    help="interval of epochs for box length updates (default 0 meaning no update)"
                    )
parser.add_argument("--relax_therm", type=int, default=5, 
                    help="thermalization steps for box length updates (default 5)"
                    )
parser.add_argument("--relax_lr", type=float, default=1.0, 
                    help="learning rate for box length updates (default: 1.0)"
                    )
parser.add_argument("--relax_min_lr", type=float, default=0.1, 
                    help="minimum learning rate for box length updates"
                    )
parser.add_argument("--relax_decay", type=float, default=0.97, 
                    help="decay rate of the learning rate for box length updates"
                    )
parser.add_argument("--num_recent_vals", type=int, default=50, 
                    help="number of recent values of pressure and stress (default 50)"
                    )

# parameters of the MCMC sampler
parser.add_argument("--mc_therm", type=int, default=10, 
                    help="MCMC thermalization steps"
                    )
parser.add_argument("--mc_steps", type=int, default=200, 
                    help="MCMC update steps"
                    )
parser.add_argument("--mc_stddev", type=float, default=0.05, 
                    help="MCMC standard deviation"
                    )

# training parameters
parser.add_argument("--batch", type=int, default=1024, 
                    help="batch size"
                    )
parser.add_argument("--acc_steps", type=int, default=1, 
                    help="accumulation steps"
                    )
parser.add_argument("--epoch", type=int, default=10000, 
                    help="final epoch"
                    )
parser.add_argument("--seed", type=int, default=42, 
                    help="random key"
                    )
parser.add_argument("--num_devices", type=int, default=1, 
                    help="number of GPU devices"
                    )

# only for speflw
parser.add_argument("--state_index_level", type=int, default=0, 
                    help="base frequency state index level"
                    )

args = parser.parse_args()

####################################################################################################

if args.executor == "train":
    
    from executor.main_train import main_train
    args_train = argparse.Namespace(
        #========== params savedata ==========
        folder=args.folder,
        load_ckpt=args.load_ckpt,
        ckpt_epochs=args.ckpt_epochs,
        #========== params physical==========
        isotope=args.isotope,
        dpfile=args.dpfile,
        ensemble=args.ensemble,
        target_pressure=args.target_pressure,
        volume_per_atom=args.volume_per_atom,
        supercell_length=args.supercell_length,
        supercell_size=args.supercell_size,
        lattice_type=args.lattice_type,
        coordinate_type=args.coordinate_type,
        dim=args.dim,
        hessian_type=args.hessian_type,
        #========== params autoregressive ==========
        temperature=args.temperature,
        num_levels=args.num_levels,
        indices_group=args.indices_group,
        van_type=args.van_type,
        van_layers=args.van_layers,
        van_size=args.van_size,
        van_heads=args.van_heads,
        van_hidden=args.van_hidden,
        #========== params flow ==========
        flow_type=args.flow_type,
        flow_depth=args.flow_depth,
        mlp_width=args.mlp_width,
        mlp_depth=args.mlp_depth,
        flow_st=args.flow_st,
        #========== params optimizer ==========
        lr_class=args.lr_class,
        lr_quant=args.lr_quant,
        min_lr_class=args.min_lr_class,
        min_lr_quant=args.min_lr_quant,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        decay_begin=args.decay_begin,
        hutchinson=args.hutchinson,
        clip_factor=args.clip_factor,
        cal_stress=args.cal_stress,
        #========== params relaxation ==========
        relax_begin=args.relax_begin,
        relax_steps=args.relax_steps,
        relax_therm=args.relax_therm,
        relax_lr=args.relax_lr,
        relax_min_lr=args.relax_min_lr,
        relax_decay=args.relax_decay,
        num_recent_vals=args.num_recent_vals,
        #========== params thermal ==========
        mc_therm=args.mc_therm,
        mc_steps=args.mc_steps,
        mc_stddev=args.mc_stddev,
        #========== params training ==========
        batch=args.batch,
        acc_steps=args.acc_steps,
        epoch=args.epoch,
        num_devices=args.num_devices,
        seed=args.seed,
        )
    main_train(args_train)

####################################################################################################

elif args.executor == "spehar":
    
    from executor.main_spehar import main_spehar
    args_spehar = argparse.Namespace(
        #========== params savedata ==========
        folder=args.folder,
        #========== params physical==========
        isotope=args.isotope,
        dpfile=args.dpfile,
        ensemble=args.ensemble,
        target_pressure=args.target_pressure,
        volume_per_atom=args.volume_per_atom,
        supercell_length=args.supercell_length,
        supercell_size=args.supercell_size,
        lattice_type=args.lattice_type,
        coordinate_type=args.coordinate_type,
        dim=args.dim,
        hessian_type=args.hessian_type,
        seed=args.seed,
        )
    main_spehar(args_spehar)
    
####################################################################################################

elif args.executor == "speflw":
    
    from executor.main_speflw import main_speflw
    args_speflw = argparse.Namespace(
        #========== params savedata ==========
        load_ckpt=args.load_ckpt,
        #========== params physical==========
        state_index_level=args.state_index_level,
        hessian_type=args.hessian_type,
        hutchinson=args.hutchinson,
        mc_therm=args.mc_therm,
        mc_steps=args.mc_steps,
        mc_stddev=args.mc_stddev,  
        batch=args.batch,
        acc_steps=args.acc_steps,
        num_devices=args.num_devices,
        seed=args.seed,
        )
    main_speflw(args_speflw)
    
####################################################################################################

elif args.executor == "speflwvan":
    
    from executor.main_speflwvan import main_speflwvan
    args_speflw = argparse.Namespace(
        #========== params savedata ==========
        load_ckpt=args.load_ckpt,
        #========== params physical==========
        hessian_type=args.hessian_type,
        hutchinson=args.hutchinson,
        mc_therm=args.mc_therm,
        mc_steps=args.mc_steps,
        mc_stddev=args.mc_stddev,  
        batch=args.batch,
        acc_steps=args.acc_steps,
        epoch=args.epoch,
        num_devices=args.num_devices,
        seed=args.seed,
        )
    main_speflwvan(args_speflw)
    
