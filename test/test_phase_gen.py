import itertools
from typing import Type

import numpy as np
import pytest
import torch

from mp_pytorch.phase_gn import PhaseGenerator, LinearPhaseGenerator, ExpDecayPhaseGenerator

DTYPES = [torch.float16, torch.float32, torch.float64]
ALL_PHASE_GEN = [LinearPhaseGenerator, ExpDecayPhaseGenerator]


@pytest.mark.parametrize('gen_type', ALL_PHASE_GEN)
@pytest.mark.parametrize('tau', [1e-5, 0.5, 1, 2.5, 5])
@pytest.mark.parametrize('delay', [0, 0.5, 1, 2.5, 5])
@pytest.mark.parametrize('dtype', DTYPES)
def test_feature_bounds(gen_type: Type[PhaseGenerator], tau: float, delay: float, dtype: torch.dtype):
    generator = gen_type(tau=tau, delay=delay, dtype=dtype)
    times = torch.arange(0, 5, 0.5)
    features = generator.phase(times)
    assert torch.all(features <= 1)
    assert torch.all(features >= 0)


@pytest.mark.parametrize('gen_type', ALL_PHASE_GEN)
@pytest.mark.parametrize('tau', [0, -1, np.inf, -np.inf])
@pytest.mark.parametrize('delay', [-1, np.inf, -np.inf])
@pytest.mark.parametrize('dtype', DTYPES)
def test_invalid(gen_type: Type[PhaseGenerator], tau: float, delay: float, dtype: torch.dtype):
    with pytest.raises(ValueError):
        gen_type(tau=tau, delay=delay, dtype=dtype)


@pytest.mark.parametrize('gen_type', ALL_PHASE_GEN)
@pytest.mark.parametrize('tau', [1e-5, 0.5, 1, 2.5, 5])
@pytest.mark.parametrize('delay', [0, 0.5, 1, 2.5, 5])
@pytest.mark.parametrize('dtype', DTYPES)
def test_phase(gen_type: Type[PhaseGenerator], tau: float, delay: float, dtype: torch.dtype):
    generator = gen_type(tau=tau, delay=delay, dtype=dtype)
    times = torch.arange(0, 5, 0.5)
    features = generator.unbound_phase(times)
    times_after = generator.phase_to_time(features)
    assert torch.allclose(times_after, times)


@pytest.mark.parametrize('gen_type', ALL_PHASE_GEN)
@pytest.mark.parametrize('learn_tau', [True, False])
@pytest.mark.parametrize('learn_delay', [True, False])
@pytest.mark.parametrize('dtype', DTYPES)
def test_learnable_params(gen_type: Type[PhaseGenerator], learn_tau: bool, learn_delay: bool, dtype: torch.dtype):
    generator = gen_type(learn_tau=learn_tau, learn_delay=learn_delay, dtype=dtype)
    assert generator.num_params == int(learn_tau) + int(learn_delay)


@pytest.mark.parametrize('gen_type', ALL_PHASE_GEN)
@pytest.mark.parametrize('learn_tau', [True, False])
@pytest.mark.parametrize('learn_delay', [True, False])
@pytest.mark.parametrize('params', itertools.product([1e-5, 1], repeat=2))
@pytest.mark.parametrize('dtype', DTYPES)
def test_set_learnable_params(gen_type: Type[PhaseGenerator], learn_tau: bool, learn_delay: bool, params: list,
                              dtype: torch.dtype):
    generator = gen_type(learn_tau=learn_tau, learn_delay=learn_delay, dtype=dtype)
    params = torch.as_tensor(params, dtype=dtype)
    n_learnable_values = int(learn_tau) + int(learn_delay)

    remaining_params = generator.set_params(params)
    assert torch.all(remaining_params == params[n_learnable_values:])

    current_params = generator.get_params()
    assert len(current_params) == n_learnable_values
    assert torch.all(current_params == params[:n_learnable_values])


@pytest.mark.parametrize('gen_type', ALL_PHASE_GEN)
@pytest.mark.parametrize('params', itertools.product([-1, 0.5], repeat=2))
@pytest.mark.parametrize('dtype', DTYPES)
def test_set_invalid_learnable_params(gen_type: Type[PhaseGenerator], params: list, dtype: torch.dtype):
    generator = gen_type(learn_tau=True, learn_delay=True, dtype=dtype)
    params = torch.as_tensor(params, dtype=dtype)
    if torch.all(params == 0.5):
        pytest.skip('No failure case.')
    with pytest.raises(AssertionError):
        generator.set_params(params)


@pytest.mark.parametrize('learn_tau', [True, False])
@pytest.mark.parametrize('learn_delay', [True, False])
@pytest.mark.parametrize('learn_alpha', [True, False])
@pytest.mark.parametrize('dtype', DTYPES)
def test_exp_learnable_params(learn_tau: bool, learn_delay: bool, learn_alpha: bool, dtype: torch.dtype):
    generator = ExpDecayPhaseGenerator(learn_tau=learn_tau, learn_delay=learn_delay, learn_alpha_phase=learn_alpha,
                                       dtype=dtype)
    assert generator.num_params == int(learn_tau) + int(learn_delay) + int(learn_alpha)


@pytest.mark.parametrize('learn_tau', [True, False])
@pytest.mark.parametrize('learn_delay', [True, False])
@pytest.mark.parametrize('learn_alpha', [True, False])
@pytest.mark.parametrize('params', itertools.product([1e-5, 1], repeat=3))
@pytest.mark.parametrize('dtype', DTYPES)
def test_exp_set_learnable_params(learn_tau: bool, learn_delay: bool, learn_alpha: bool, params: list,
                                  dtype: torch.dtype):
    generator = ExpDecayPhaseGenerator(learn_tau=learn_tau, learn_delay=learn_delay, learn_alpha_phase=learn_alpha,
                                       dtype=dtype)
    params = torch.as_tensor(params, dtype=dtype)
    n_learnable_values = int(learn_tau) + int(learn_delay) + int(learn_alpha)

    remaining_params = generator.set_params(params)
    assert torch.all(remaining_params == params[n_learnable_values:])

    current_params = generator.get_params()
    assert len(current_params) == n_learnable_values
    assert torch.all(current_params == params[:n_learnable_values])


@pytest.mark.parametrize('params', itertools.product([-1, 0.5], repeat=3))
@pytest.mark.parametrize('dtype', DTYPES)
def test_exp_set_invalid_learnable_params(params: list, dtype: torch.dtype):
    generator = ExpDecayPhaseGenerator(learn_tau=True, learn_delay=True, learn_alpha_phase=True, dtype=dtype)
    params = torch.as_tensor(params, dtype=dtype)
    if torch.all(params == 0.5):
        pytest.skip('No failure case.')
    with pytest.raises(AssertionError):
        generator.set_params(params)
