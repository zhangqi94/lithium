# Neural Canonical Transformations for Quantum Anharmonic Solids of Lithium

This repository provides neural canonical transformations (NCT) for investigating lithium crystals, focusing on anharmonic quantum effects.
NCT is an *ab initio* variational free energy method based on deep generative models. 
The method uses a product spectrum ansatz or a variational autoregressive network to represent phonon energy occupation probabilities (phonon Boltzmann distribution), 
a normalizing flow for phonon wave functions, 
and a deep potential model to describe the Born-Oppenheimer energy surface.

## Getting Started

### Requirements
For optimal performance, **GPUs** are recommended. Using only CPU may result in very slow computations.
Ensure your environment meets the following specifications:
- python >= 3.12
- [jax](https://github.com/jax-ml/jax) >= 0.4.31
- [optax](https://github.com/google-deepmind/optax) >= 0.2.3 
- [flax](https://github.com/google/flax) >= 0.8.5

### Training

To view the available parameters for Neural Canonical Transformations, run:
```
python3 main.py --help
```

**Basic example for training the NCT model:**
```
python3 main.py  --executor "train"  \
            --seed 33  --folder "data/"  --load_ckpt "None"  --ckpt_epochs 1000  \
            --isotope "Li"  --dpfile "dp4"  --supercell_size 2 2 2  --lattice_type "bcc"  \
            --coordinate_type "phon"  --temperature 50  \
            --indices_group 1  --num_levels 20  --lr_class 2e-2  \
            --lr_quant 5e-4  --min_lr_class 5e-4  --min_lr_quant 5e-6  \
            --decay_rate 0.97  --decay_steps 100  --decay_begin 1000  \
            --hutchinson  --hessian_type "for1"  \
            --mc_therm 10  --mc_steps 3000  --mc_stddev 0.03  --batch 400  \
            --acc_steps 1  --epoch 20000  \
            --flow_type "flw_rnvpflax"  --flow_depth 6  --mlp_width 128  --mlp_depth 2  \
            --flow_st  "st"  --van_type "van_probtab"  --ensemble "npt"  \
            --target_pressure 0.0  --volume_per_atom 19.515776  \
            --supercell_length -1 -1 -1  --relax_begin 1000  --relax_steps 50  \
            --relax_therm 5  --relax_lr 0.200  --relax_min_lr 0.010  --relax_decay 0.99  \
            --num_recent_vals 20  --cal_stress 1 
```

**Example of calculating single-phonon excitations:**
```
python3 main.py  --executor "speflw"  \
        --state_index_level 1  --seed 43  --load_ckpt "xxx/epoch_015000.pkl"  \
        --hessian_type "for1"  --hutchinson  \
        --mc_therm 4  --mc_steps 3000  --mc_stddev 0.03  --batch 512  --acc_steps 16   
```

### Code Structure

- `executor/`: Contains the main programs for neural canonical transformations.
    - `main_train.py`: Used for training the NCT model.
    - `main_speflw.py`: Used for calculating single-phonon excitations.
- `src/structures/`: Contains initial lithium crystal structures in VASP format.
    -  including *bcc*, *fcc*, *cI16*, *oC88* and *oC40* structures.
- `src/dpmodel/`: Contains [Deep Potential](https://github.com/deepmodeling/deepmd-kit) models for the Born-Oppenheimer energy surface.
    - `fznp_dp0.pkl`: Li-DP-Hyb2 model is available for high-pressure structures: *cI16* *oC88* *oC40*.
    - `fznp_dp4.pkl`: Li-DP-Hyb3 model is trained with additional *bcc* and *fcc* structures.
- `run/`: Contains scripts for job submission and hyperparameter configuration for neural canonical transformations.
- `nctstruct/`: Output strucutres (*cI16* and *oC88*) of NCT calculations at 70 GPa and 100 K.
- `dftfiles/`: Density functional theory input files for the PBE and HSE06 functionals, specifically for [ABACUS](https://github.com/abacusmodeling/abacus-develop).

## Citation

If you use this code in your research, please cite:
```
@article{zhang2025neural,
  title = {Neural Canonical Transformations for Quantum Anharmonic Solids of Lithium},
  author = {Zhang, Qi and Wang, Xiaoyang and Shi, Rong and Ren, Xinguo and Wang, Han and Wang, Lei},
  journal = {Phys. Rev. Lett.},
  volume = {134},
  issue = {24},
  pages = {246101},
  numpages = {7},
  year = {2025},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/p3th-25bc},
  url = {https://link.aps.org/doi/10.1103/p3th-25bc}
}
```

For NCT applied to molecular vibrations, please refer to the GitHub repository: [https://github.com/zhangqi94/VibrationalSystem](https://github.com/zhangqi94/VibrationalSystem)
The associated paper is:
```
@article{zhang2024neural,
    author = {Zhang, Qi and Wang, Rui-Si and Wang, Lei},
    title = "{Neural canonical transformations for vibrational spectra of molecules}",
    journal = {The Journal of Chemical Physics},
    volume = {161},
    number = {2},
    pages = {024103},
    year = {2024},
    month = {07},
    issn = {0021-9606},
    doi = {10.1063/5.0209255},
    url = {https://doi.org/10.1063/5.0209255},
    eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0209255/20033705/024103\_1\_5.0209255.pdf},
}
```

