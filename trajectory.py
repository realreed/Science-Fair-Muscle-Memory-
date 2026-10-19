import numpy as np

def generate_trajectory(start_angles, end_angles, steps=120):
    start = np.array(start_angles, dtype=float)
    end = np.array(end_angles, dtype=float)

    trajectory = np.linspace(start, end, steps)
    return trajectory