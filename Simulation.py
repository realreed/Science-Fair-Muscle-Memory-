from pathlib import Path
import argparse
import time

import pybullet as p
import pybullet_data

from robot.controller import move_to_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Run without a window or animation delays.")
    args = parser.parse_args()
    p.connect(p.DIRECT if args.fast else p.GUI)

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setRealTimeSimulation(0)

        p.loadURDF("plane.urdf")


        arm_path = Path(__file__).resolve().parent / "robot" / "arm.URDF"

        robot = p.loadURDF(
            str(arm_path),
            basePosition=[0, 0, 0.5],
            useFixedBase=True,
        )


        for joint in [0, 1]:
            p.resetJointState(robot, joint, targetValue=0)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=45,
            cameraPitch=-55,
            cameraTargetPosition=[0.3, 0, 0.5],
        )

        target_x = 0.5
        target_y = 0.3

        print("Running one baseline movement...")
        print(f"Target: ({target_x}, {target_y}, 0.55)")

        end_position, error, planning_seconds, memory_used = move_to_target(
            robot,
            target_x,
            target_y,
            mode="normal",
            realtime=not args.fast,
        )

        print("End position:", end_position)
        print(f"Movement error: {error:.6f} meters")
        print("Success:", error < 0.01)
        print(f"Planning time: {planning_seconds * 1000:.4f} ms")
        print("Memory used:", memory_used)
        if args.fast:
            return
        print("Close the simulation window when finished.")


        while p.isConnected():
            p.stepSimulation()
            time.sleep(1 / 240)

    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
