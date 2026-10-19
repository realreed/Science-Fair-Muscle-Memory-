"""Paired trials: 5 rounds x 150 targets x 2 models = 1,500 movements."""
import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

from robot.controller import move_to_target
from robot.memory_stuff import memory_bank, get_memory_count

ROOT = Path(__file__).resolve().parent


def make_targets(seed, count):
    """Sample feasible joint poses, then convert to Cartesian targets.

    This defines a reproducible task distribution, not uniform Cartesian sampling.
    Neither model sees the sampled joint angles.
    """
    rng = np.random.default_rng(seed)
    shoulder = rng.uniform(-0.8, 0.8, count)
    elbow = rng.uniform(0.6, 2.2, count)
    x = 0.5 * np.cos(shoulder) + 0.5 * np.cos(shoulder + elbow)
    y = 0.5 * np.sin(shoulder) + 0.5 * np.sin(shoulder + elbow)
    return list(zip(x, y))


def run(rounds, pairs, seed, output):
    output.mkdir(parents=True, exist_ok=True)
    p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setRealTimeSimulation(0)
        p.loadURDF('plane.urdf')
        robot = p.loadURDF(str(ROOT / 'robot' / 'arm.URDF'),
                           basePosition=[0, 0, 0.5], useFixedBase=True)
        for joint in (0, 1):
            p.resetJointState(robot, joint, 0, targetVelocity=0)
        initial_state = p.saveState()

        # Untimed warm-up is excluded from data and memory is cleared afterward.
        for mode in ('normal', 'memory'):
            p.restoreState(initial_state)
            move_to_target(robot, 0.5, 0.3, mode=mode, realtime=False)
        memory_bank.clear()

        fields = ['round', 'pair', 'order_in_pair', 'mode', 'target_x', 'target_y',
                  'planning_seconds', 'planning_cpu_seconds', 'runtime_seconds',
                  'runtime_cpu_seconds', 'error_m', 'success', 'memory_used',
                  'memory_before', 'memory_after', 'fresh_trajectory_generations',
                  'trajectory_points', 'simulation_steps']
        with (output / 'trials.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for round_number in range(1, rounds + 1):
                memory_bank.clear()
                targets = make_targets(seed + round_number, pairs)
                for pair, (x, y) in enumerate(targets, start=1):
                    # Alternate execution order to reduce systematic timing bias.
                    modes = ('normal', 'memory') if (pair + round_number) % 2 else ('memory', 'normal')
                    for order, mode in enumerate(modes, start=1):
                        p.restoreState(initial_state)
                        before = get_memory_count()
                        metrics = {}
                        wall_start = time.perf_counter()
                        cpu_start = time.process_time()
                        _, error, planning, reused = move_to_target(
                            robot, float(x), float(y), mode=mode,
                            realtime=False, metrics=metrics)
                        cpu_elapsed = time.process_time() - cpu_start
                        elapsed = time.perf_counter() - wall_start
                        writer.writerow(dict(
                            round=round_number, pair=pair, order_in_pair=order,
                            mode=mode, target_x=x, target_y=y,
                            planning_seconds=planning, runtime_seconds=elapsed,
                            runtime_cpu_seconds=cpu_elapsed, error_m=error,
                            success=error < 0.01, memory_used=reused,
                            memory_before=before, memory_after=get_memory_count(),
                            **metrics))
                    handle.flush()
                print(f'Round {round_number}/{rounds} complete: {pairs * 2} movements', flush=True)
        p.removeState(initial_state)
    finally:
        memory_bank.clear()
        p.disconnect()

    metadata = dict(rounds=rounds, targets_per_round=pairs,
                    recorded_movements=rounds * pairs * 2, seed=seed,
                    python=sys.version, platform=platform.platform(),
                    pybullet_api=p.getAPIVersion(), timestep_seconds=1 / 240,
                    trajectory_points=120, settling_steps=120,
                    success_threshold_m=0.01, target_tolerance_m=0.10,
                    start_tolerance_joint_norm_radians=0.10,
                    memory_reset='Between rounds; grows within each round',
                    starting_pose='Zero joint angles and velocities; restore full physics state every movement',
                    tasks='Uniform shoulder [-0.8, 0.8], elbow [0.6, 2.2] radians; forward kinematics to targets',
                    warmup='One movement per model, excluded from all recorded trials',
                    planning='Inverse kinematics plus memory lookup and trajectory generation/adaptation',
                    runtime='Controller call, including execution and memory storage; excludes physics reset and CSV writing',
                    count_metric='Fresh trajectory generations, not internal IK iterations or reasoning steps',
                    limitation='Both models still calculate inverse kinematics; reuse does not eliminate that computation.')
    (output / 'settings.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved {rounds * pairs * 2} movements to {output / "trials.csv"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--pairs', type=int, default=150)
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.rounds < 1 or args.pairs < 1:
        parser.error('rounds and pairs must be positive')
    run(args.rounds, args.pairs, args.seed, args.output)
