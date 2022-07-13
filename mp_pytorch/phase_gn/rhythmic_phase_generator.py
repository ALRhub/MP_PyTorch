import numpy as np

from mp_pytorch import PhaseGenerator


# TODO: Adjust to mp_pytorch
class RhythmicPhaseGenerator(PhaseGenerator):

    def phase(self, t: np.ndarray, duration: float):
        linear_phase = t / duration
        phase = linear_phase % 1.0

        return phase
