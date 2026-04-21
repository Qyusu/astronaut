import numpy as np


def cost_function(final_energy: float, exact_energy: float) -> float:
    return np.abs(final_energy - exact_energy)
