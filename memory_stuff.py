
import numpy as np

memory_bank = {}

def save_primitive(start_angles, target, trajectory, error):
    primitive = {
        "start_angles": list(start_angles),
        "target": list(target),
        "trajectory": np.asarray(trajectory).tolist(),
        "error": float(error)
    }

    memory_bank[len(memory_bank)] = primitive
    


def find_primitive(
    start_angles,
    target,
    target_tolerance=0.10,
    start_tolerance=0.10,
):
    best_primitive = None
    best_score = float("inf")

    for primitive in memory_bank.values():
        target_distance = np.linalg.norm(
            np.array(target) - np.array(primitive["target"])
        )

        start_distance = np.linalg.norm(
            np.array(start_angles)
            - np.array(primitive["start_angles"])
        )

        if target_distance > target_tolerance:
            continue

        if start_distance > start_tolerance:
            continue

        score = (
            target_distance / target_tolerance
            + start_distance / start_tolerance
        )

        if score < best_score:
            best_score = score
            best_primitive = primitive

    return best_primitive


def get_memory_count():

    memory_count = len(memory_bank)
    return memory_count
    

def adapt_trajectory(primitive, new_start_angles, new_end_angles):
    
    
    old_start_angles = np.array(primitive["start_angles"])
    
    old_trajectory = np.array(primitive["trajectory"])
    old_end_angles = old_trajectory[-1]

    start_diff = np.array(new_start_angles) - old_start_angles
    end_diff = np.array(new_end_angles) - old_end_angles
    
    weights = np.linspace(0, 1, len(old_trajectory))
    
    adapted_trajectory = []

    for i, alpha in enumerate(weights):
        start_correction = (1 - alpha) * start_diff
        end_correction = alpha * end_diff

        adapted_point = old_trajectory[i] + start_correction + end_correction

        adapted_trajectory.append(adapted_point.tolist())

    return adapted_trajectory

