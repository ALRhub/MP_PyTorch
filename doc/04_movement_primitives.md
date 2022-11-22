# 4. Movement Primitives
Currently, we provide three Movement Primitives, including Dynamic Movement Primitives(DMPs), Probablistic Movement Primitives(ProMPs) and Probablistic Dynamic Movement Primitives(ProDMPs).
We also provide two Movement Primitives Interfaces(Dynamic based and Probablistic based), which can be used to define the custom Movement Primitives. 
You can register your custom MPs to the MPFactory and create the MP instance given configuration.

The main features of the MP Factory and MP Interfaces are as follows:

| Classes                  | Main Functions                           | Description                                                                                   |
|--------------------------|------------------------------------------|-----------------------------------------------------------------------------------------------|
| MPFactory                | `MPFactory.init_mp`                      | Create an MP instance given configuration.                                                    |
| MPInterface              |                                          | Abstract Basic Class for Deterministic Movement Primitives                                    |
|                          | `MPInterface.update_inputs`              | Update MPs parameters                                                                         |
|                          | `MPInterface.get_trajs`                  | Get movement primitives trajectories given flag                                               |
|                          | `MPInterface.learn_mp_params_from_trajs` | Abstractmethod for learning parameters from trajectories                                      |
| ProbabilisticMPInterface |                                          | Abstract Basic Class for Probablistic Movement Primitives                                     |
|                          | `MPInterface.update_inputs`              | Update MPs parameters                                                                         |
|                          | `MPInterface.get_trajs`                  | Get movement primitives trajectories given flag, including trajectories mean and distribution |
|                          | `MPInterface.sample_trajectories`        | Sample trajectories from MPs                                                                  |
|                          | `MPInterface.learn_mp_params_from_trajs` | Abstractmethod for learning parameters from trajectories                                      |

&nbsp;
### 4.1 Dynamic Movement Primitives
We provide a [DMPs demo](../mp_pytorch/demo/demo_dmp.py) to show how to create a DMPs instance and visualize the corresponding result.

To run the demo, you can run the following code:
```python
from mp_pytorch import demo
demo.test_dmp()
```

&nbsp;
### 4.2 Probablistic Movement Primitives
We provide a [ProMPs demo](../mp_pytorch/demo/demo_promp.py) to show how to create a ProMPs instance and visualize the corresponding result.

To run the demo, you can run the following code:
```python
from mp_pytorch import demo
demo.test_promp()
demo.test_zero_padding_promp()
```

&nbsp;
### 4.3 Probablistic Dynamic Movement Primitives
[Probablistic Dynamic Movement Primitives(ProDMPs)](https://arxiv.org/abs/2210.01531) is a recently presented Method, which combing the Dynamic and Probablistic properties of Movement Primitives from a unified perspective.

We provide a [ProDMPs demo](../mp_pytorch/demo/demo_prodmp.py) to show how to create a ProDMPs instance and visualize the corresponding result.
To run the demo, you can run the following code:
```python
from mp_pytorch import demo
demo.test_prodmp()
```


[Back to Overview](./)
