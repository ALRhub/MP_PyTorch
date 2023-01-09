# 2. Edit Configuration

We recommend you using `addict` or `yaml` to edit the configuration files.

&nbsp;
### 2.1 Demo
We provide a [demo](../mp_pytorch/demo/demo_mp_config.py) to show how to edit the configuration. 

You can call this demo as follows:

```python
from mp_pytorch import demo
config, times, params, params_L, init_time, init_pos, init_vel, demos = \
    demo.get_mp_utils(mp_type="prodmp", learn_tau=True, learn_delay=True)
```

&nbsp;
### 2.2 Parameters in Configuration

| Type                | Parameters               | Description                 |
|---------------------|--------------------------|-----------------------------|
| General             | `num_dof`                | Number of DoFs              |
|                     | `tau`                    |                             |
|                     | `learn_tau`              | If tau is learnable         | 
|                     | `learn_delay`            | If delay is learnable       |
| Movement Primitives | `num_basis`              | Number of Basis functions   |
|                     | `basis_bandwidth_factor` |                             |
|                     | `alpha`                  |                             |
|                     | `alpha_phase`            |                             |
|                     | `dt`                     | Timestep                    |
|                     | `weights_scale`          |                             |
|                     | `goal_scale`             |                             |
|                     | `mp_type`                | Type of Movement Primitives |


[Back to Overview](./)