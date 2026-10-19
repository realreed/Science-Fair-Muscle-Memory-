import math
import time

import numpy as np
import pybullet as p

from robot.trajectory import generate_trajectory
from robot.memory_stuff import (
    save_primitive,
    find_primitive,
    adapt_trajectory,
)


def move_to_target(robot, x, y, mode="normal", realtime=True, metrics=None):
    if mode not in ("normal", "memory"):
        raise ValueError(f"Unknown mode: {mode}")

    target = [x, y]
    target_position = [x, y, 0.55]

    start_angles = [
        p.getJointState(robot, 0)[0],
        p.getJointState(robot, 1)[0],
    ]

    planning_start = time.perf_counter()
    planning_cpu_start = time.process_time()
    end_angles = p.calculateInverseKinematics(
        robot,
        2,
        target_position,
        maxNumIterations=1000,
        residualThreshold=0.00001,
    )[:2]

    if end_angles is None:
        raise ValueError(f"Target {target} is unreachable for this robot arm.")

    end_angles = end_angles[:2]
    if len(end_angles) != 2:
        raise ValueError(f"Inverse kinematics returned invalid angles: {end_angles}")
    memory_used = False
    if mode == "memory":
        primitive = find_primitive(start_angles, target)

        if primitive is not None:
            trajectory = adapt_trajectory(
                primitive,
                start_angles,
                end_angles,
            )
            memory_used = True
        else:
            trajectory = generate_trajectory(
                start_angles,
                end_angles,
            )
    else:
        trajectory = generate_trajectory(
            start_angles,
            end_angles,
        )
    planning_seconds = time.perf_counter() - planning_start
    planning_cpu_seconds = time.process_time() - planning_cpu_start
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.size == 0:
        raise ValueError("Generated trajectory is empty.")
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(1, -1)

    for angles in trajectory:
        joint_angles = [float(angle) for angle in np.atleast_1d(angles)]
        if len(joint_angles) != 2:
            raise ValueError(f"Trajectory contains invalid joint angles: {angles}")

        p.setJointMotorControlArray(
            robot,
            [0, 1],
            p.POSITION_CONTROL,
            targetPositions=joint_angles,
            forces=[100, 80],
        )

        p.stepSimulation()
        if realtime:
            time.sleep(1 / 240)

    for _ in range(120):
        p.stepSimulation()
        if realtime:
            time.sleep(1 / 240)

    end_effector_pos = p.getLinkState(
        robot,
        2,
        computeForwardKinematics=True,
    )[4]

    error = math.dist(target_position, end_effector_pos)

    if mode == "memory" and error < 0.01:
        save_primitive(
            start_angles,
            target,
            trajectory,
            error,
        )

    if metrics is not None:
        metrics.update(
            planning_cpu_seconds=planning_cpu_seconds,
            fresh_trajectory_generations=int(not memory_used),
            trajectory_points=len(trajectory),
            simulation_steps=len(trajectory) + 120,
        )

    return end_effector_pos, error, planning_seconds, memory_used
