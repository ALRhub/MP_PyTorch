# MP_PyTorch: The Movement Primitives Package in PyTorch

MP_PyTorch package focus on **Movement Primitives(MPs) on Imitation Learning(IL) and Reinforcement Learning(RL)** and provides convenient movement primitives interface implemented by PyTorch, including DMPs, ProMPs and [ProDMPs](https://arxiv.org/abs/2210.01531). 
Users can also implement custom Movement Primitives according to the basis and phase generator. Further, advanced NN-based Movement Primitives Algorithm can also be realized according to the convenient PyTorch-based Interface.
This package aims to building a movement primitives toolkit which could be combined with modern imitation learning and reinforcement learning algorithm.  

<!--
## Dependencies:
pytorch, addict, numpy, matplotlib
-->

&nbsp;
## Installation

For the installation we recommend you set up a conda environment or venv beforehand. 

This package will automatically install the following dependencies: addict, numpy, pytorch and matplotlib.

### 1. Install from PyPI
```bash
pip install mp_pytorch
```

### 2. Install from source

```bash 
git clone git@github.com:ALRhub/MP_PyTorch.git
cd mp_pytorch
pip install -e .
```

After installation, you can import the package easily.
```bash
import mp_pytorch
from mp_pytorch import MPFactory
```

&nbsp;
## Quickstart
For further information, please refer to the [User Guide](./doc/README.md).

The main steps to create ProDMPs instance and generate trajectories are as follows:

### 1. Edit configuration 
Suppose you have edited the required configuration.
You can view the demo and check how to edit the configuration in [Edit Configuration](./doc/02_config.md).
```python
# config, times, params, params_L, init_time, init_pos, init_vel, demos = get_mp_utils("prodmp", True, True)
```

### 2. Initial prodmp instance and update inputs
```python
mp = MPFactory.init_mp(**config)
mp.update_inputs(times=times, params=params, params_L=params_L,
                 init_time=init_time, init_pos=init_pos, init_vel=init_vel)

# you can also choose to learn parameters from demonstrations.
params_dict = mp.learn_mp_params_from_trajs(times, demos)
```

### 3. Generate trajectories
```python
traj_dict = mp.get_trajs(get_pos=True, get_pos_cov=True,
                         get_pos_std=True, get_vel=True,
                         get_vel_cov=True, get_vel_std=True)

# for probablistic movement primitives, you can also choose to sample trajectories
samples, samples_vel = mp.sample_trajectories(num_smp=10)
```

The structure of this package can be seen as follows:

| Types                   | Classes                                  | Description                                                                  |
|-------------------------|------------------------------------------|------------------------------------------------------------------------------|
| **Phase Generator**     | `PhaseGenerator`                         | Interface for Phase Generators                                               |       
|                         | `RhythmicPhaseGenerator`                 | Rhythmic phase generator                                                     |       
|                         | `SmoothPhaseGenerator`                   | Smooth phase generator                                                       |       
|                         | `LinearPhaseGenerator`                   | Linear phase generator                                                       |       
|                         | `ExpDecayPhaseGenerator`                 | Exponential decay phase generator                                            |       
| **Basis Generator**     | `BasisGenerator`                         | Interface for Basis Generators                                               |       
|                         | `RhythmicBasisGenerator`                 | Rhythmic basis generator                                                     |       
|                         | `NormalizedRBFBasisGenerator`            | Normalized RBF basis generator                                               |       
|                         | `ProDMPBasisGenerator`                   | ProDMP basis generator                                                       |       
| **Movement Primitives** | `MPFactory`                              | Create an MP instance given configuration                                    |       
|                         | `MPInterface`                            | Interface for Deterministic Movement Primitives                              |       
|                         | `ProbabilisticMPInterface`               | Interface for Probablistic Movement Primitives                               |        
|                         | `DMP`                                    | Dynamic Movement Primitives                                                  |       
|                         | `ProMP`                                  | Probablistic Movement Primitives                                             |        
|                         | `ProDMP`                                 | [Probablistic Dynamic Movement Primitives](https://arxiv.org/abs/2210.01531) |        
 
 
 
&nbsp;
## Cite
If you interest this project and use it in a scientific publication, we would appreciate citations to the following information:
```markdown
@article{li2023prodmp,
  title={ProDMP: A Unified Perspective on Dynamic and Probabilistic Movement Primitives},
  author={Li, Ge and Jin, Zeqi and Volpp, Michael and Otto, Fabian and Lioutikov, Rudolf and Neumann, Gerhard},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}

```

&nbsp;
## Team
MP_PyTorch is developed and maintained by the [ALR-Lab](https://alr.anthropomatik.kit.edu)(Autonomous Learning Robots Lab), KIT. 


<!DOCTYPE html>
<html>
  <head>    
    <meta name="google-site-verification" content="TBTpxqGVKOpnljA1-tH3WLxVPTqhX4y3d1voyDE_BSE" />
  </head>
  <body>
    <p>Welcome to our GitHub Pages!</p>
  </body>
</html>

