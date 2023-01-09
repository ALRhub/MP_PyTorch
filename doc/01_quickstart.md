# 1. Quick Start

MP_PyTorch provides convenient interfaces to develop Movement Primitives with modern Imitation Learning and Reinforcement Learning algorithm. 
You can create the basic Moment Primitives Instance(DMPs, ProMPs and ProDMPs) in the MPFactory or define your own custom MPs with the MPInterface.
It's also convenient to combine the MPs with modern neural networks based algorithm to realize more complex task.

&nbsp;
### 1.1 Quick start for MPFactory
In this quick start section, we will provide a demo showing how to create ProDMPs instance and generate trajectories.

#### 1.1.1 Edit Configuration 
Suppose you have edited the required configuration.
You can view the demo and check how to edit the configuration in [Edit Configuration](./02_config.md).
```python
# config, times, params, params_L, init_time, init_pos, init_vel, demos = get_mp_utils("prodmp", True, True)
```

#### 1.1.2 Initial ProDMPs instance and update inputs
```python
mp = MPFactory.init_mp(**config)
mp.update_inputs(times=times, params=params, params_L=params_L,
                 init_time=init_time, init_pos=init_pos, init_vel=init_vel)

# you can also choose to learn parameters from demonstrations.
params_dict = mp.learn_mp_params_from_trajs(times, demos)
```

#### 1.1.3 Generate trajectories
```python
traj_dict = mp.get_trajs(get_pos=True, get_pos_cov=True,
                         get_pos_std=True, get_vel=True,
                         get_vel_cov=True, get_vel_std=True)

# for probablistic movement primitives, you can also choose to sample trajectories
samples, samples_vel = mp.sample_trajectories(num_smp=10)
```

&nbsp;
### 1.2 Define the custom Movement Primitives
To define the custom Movement Primitives method, you need to understand the following interfaces in corresponding sections:
- [Phase Generator Interface](./03_phase_and_basis.md)
- [Basis Generator Interface](./03_phase_and_basis.md)
- [Movement Primitives Interface](./04_movement_primitives.md)

&nbsp;
### 1.3 Combing Movement Primitives with Neural Networks
**The corresponding docs and demos are under construction.** 




[Back to Overview](./)