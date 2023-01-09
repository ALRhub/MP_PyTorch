# MP_PyTorch User Guide

### Introduction and Quick Start
- [1. Quick Start](./01_quickstart.md) 

### Basic Movement Primitives
- [2. Edit Configuration](./02_config.md)
- [3. Phase and Basis Generation](./03_phase_and_basis.md)
- [4. Movement Primitives](./04_movement_primitives.md)

### Combing Movement Primitives with Neural Networks
- [5. NN-based Movement Primitives](./05_nn-based_mp.md)



<!--
| Types               | Classes                                 | Main Functions                                    | Description                                                                                                   | Status |
|---------------------|-----------------------------------------|---------------------------------------------------|---------------------------------------------------------------------------------------------------------------|--------|
| phase generator     | PhaseGenerator                          |                                                   | Abstract Basic Class for Phase Generators. Transfer time duration to [0, 1] range.                            |        |
|                     |                                         | `PhaseGenerator.phase`                            | Abstractmethod for phase interface.                                                                           |        |
|                     |                                         | `PhaseGenerator.unbound_phase`                    | Abstractmethod for unbound phase interface.                                                                   |        |
|                     |                                         | `PhaseGenerator.phase_to_time`                    | Abstractmethod for inverse operation, compute times given phase.                                              |        |
|                     |                                         | `PhaseGenerator.set_params`                       | Set parameters of current object and attributes                                                               |        |
|                     |                                         | `PhaseGenerator.get_params`                       | Return all learnable parameters.                                                                              |        |
|                     |                                         | `PhaseGenerator.get_params_bounds`                | Return all learnable parameters' bounds.                                                                      |        |
|                     |                                         | `PhaseGenerator.finalize`                         | Mark the phase generator as finalized so that the parameters cannot be updated any more.                      |        |
|                     | RhythmicPhaseGenerator                  |                                                   | Rhythmic phase generator.                                                                                     |        |
|                     | SmoothPhaseGenerator                    |                                                   | Smooth phase generator with five order spline phase                                                           |        |
|                     | LinearPhaseGenerator                    |                                                   | Linear Phase Generator                                                                                        |        |
|                     | ExpDecayPhaseGenerator                  |                                                   | Exponential decay phase generator                                                                             |        |
| basis generator     | BasisGenerator                          |                                                   | Abstract Basic Class for Basis Generators                                                                     |        |
|                     |                                         | `BasisGenerator.basis`                            | Abstractmethod to generate value of single basis function at given time.                                      |        |
|                     |                                         | `BasisGenerator.basis_multi_dofs`                 | Interface to generate basis functions for multi-dof at given time                                             |        |
|                     |                                         | `BasisGenerator.set_params`                       | Set parameters of current object and attributes                                                               |        |
|                     |                                         | `BasisGenerator.get_params`                       | Return all learnable parameters                                                                               |        |
|                     |                                         | `BasisGenerator.get_params_bounds`                | Return all learnable parameters' bounds                                                                       |        |
|                     |                                         | `BasisGenerator.show_basis`                       | Visualize the basis functions for debug usage                                                                 |        |
|                     | RhythmicBasisGenerator                  |                                                   | Rhythmic Basis Generator                                                                                      |        |
|                     | NormalizedRBFBasisGenerator             |                                                   | Normalized RBF basis generator                                                                                |        |
|                     | ZeroPaddingNormalizedRBFBasisGenerator  |                                                   | Normalized RBF with zero padding basis generator                                                              |        |
|                     | ProDMPBasisGenerator                    |                                                   | ProDMP basis generator                                                                                        |        |
|                     |                                         | `ProDMP.pre_compute`                              | Precompute basis functions and other stuff.                                                                   |        |
|                     |                                         | `BasisGenerator.basis_and_phase`                  | Set basis and phase for the rhythmic basis generator                                                          |        |
|                     |                                         | `BasisGenerator.auto_compute_basis_scale_factors` | Compute scale factors for each basis function                                                                 |        |
|                     |                                         | `BasisGenerator.times_to_indices`                 | Map time points to pre-compute indices                                                                        |        |
| movement primitives | MPFactory                               | `MPFactory.init_mp`                               | Create an MP instance given configuration. You can also directly initialize the MPs without using this class. |        |
|                     | MPInterface                             |                                                   | Abstract Basic Class for Deterministic Movement Primitives                                                    |        |
|                     |                                         | `MPInterface.update_inputs`                       | Update MPs parameters                                                                                         |        |
|                     |                                         | `MPInterface.get_trajs`                           | Get movement primitives trajectories given flag                                                               |        |
|                     |                                         | `MPInterface.learn_mp_params_from_trajs`          | Abstractmethod for learning parameters from trajectories                                                      |        |
|                     | ProbabilisticMPInterface                |                                                   | Abstract Basic Class for Probablistic Movement Primitives                                                     |        |
|                     |                                         | `MPInterface.update_inputs`                       | Update MPs parameters                                                                                         |        |
|                     |                                         | `MPInterface.get_trajs`                           | Get movement primitives trajectories given flag, including trajectories mean and distribution                 |        |
|                     |                                         | `MPInterface.sample_trajectories`                 | Sample trajectories from MPs                                                                                  |        |
|                     |                                         | `MPInterface.learn_mp_params_from_trajs`          | Abstractmethod for learning parameters from trajectories                                                      |        |
|                     | DMP                                     |                                                   | Class for Dynamic Movement Primitives                                                                         |        |
|                     | ProMP                                   |                                                   | Class for Probablistic Movement Primitives                                                                    |        |
|                     | ProDMP                                  |                                                   | Class for [Probablistic Dynamic Movement Primitives](https://arxiv.org/abs/2210.01531)                        |        |
|                     |                                         | `ProDMP.set_initial_conditions`                  | Set initial conditions for ProDMP in a batched manner                                                        |        |
|                     |                                         | `ProDMP.compute_intermediate_terms_single`        | Determinant of initial condition                                                                             |        |
|                     |                                         | `ProDMP.compute_intermediate_terms_multi_dof`     | Determinant of initial condition                                                                             |        |
-->