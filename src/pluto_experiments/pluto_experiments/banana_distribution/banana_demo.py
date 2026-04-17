"""
Banana Distribution Demo — Chapter 5 Experiment.
Samples 1000 poses from velocity motion model and plots scatter.
Shows how noise parameters α1..α6 shape the distribution.
Run standalone (no ROS2 required) for quick visualization.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[5]))

from pluto_filters.motion_models.velocity_motion_model import (
    sample_motion_model_velocity, DEFAULT_ALPHAS
)


def run_banana_demo(v: float = 0.5, omega: float = 0.05, dt: float = 1.0,
                    alphas: list = None, n_samples: int = 1000,
                    save_path: str | None = None):
    """
    Generate and optionally plot the banana distribution.
    Returns array of sampled poses (n_samples, 3).
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    start_pose = np.array([0.0, 0.0, 0.0])
    samples = np.array([
        sample_motion_model_velocity(start_pose, v, omega, dt, alphas)
        for _ in range(n_samples)
    ])

    print(f"Banana Distribution: v={v:.2f} m/s, ω={omega:.3f} rad/s, dt={dt:.1f}s")
    print(f"  α = {alphas}")
    print(f"  Mean  x={samples[:,0].mean():.3f}, y={samples[:,1].mean():.3f}")
    print(f"  Std   x={samples[:,0].std():.3f},  y={samples[:,1].std():.3f}")
    print(f"  Range x=[{samples[:,0].min():.3f}, {samples[:,0].max():.3f}]")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Banana Distribution: v={v:.2f}, ω={omega:.3f}, dt={dt:.1f}s', fontsize=13)

        # XY scatter (the banana)
        axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c='#F4A700')
        axes[0].plot(0, 0, 'k^', markersize=10, label='Start')
        axes[0].set_xlabel('x [m]')
        axes[0].set_ylabel('y [m]')
        axes[0].set_title('XY Distribution (Banana)')
        axes[0].legend()
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)

        # X histogram
        axes[1].hist(samples[:, 0], bins=50, color='steelblue', alpha=0.7, edgecolor='k', lw=0.5)
        axes[1].set_xlabel('x [m]')
        axes[1].set_title('X marginal')
        axes[1].grid(True, alpha=0.3)

        # Theta histogram
        axes[2].hist(np.degrees(samples[:, 2]), bins=50, color='coral', alpha=0.7, edgecolor='k', lw=0.5)
        axes[2].set_xlabel('θ [deg]')
        axes[2].set_title('θ marginal')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
            print(f"  Saved to {save_path}")
        else:
            out = Path(__file__).parent / 'banana.png'
            plt.savefig(str(out), dpi=120)
            print(f"  Saved to {out}")
        plt.close()
    except ImportError:
        print("  (matplotlib not available, skipping plot)")

    return samples


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Banana distribution demo')
    parser.add_argument('--v', type=float, default=0.5)
    parser.add_argument('--omega', type=float, default=0.05)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()

    # Low noise
    print("\n--- Low noise ---")
    run_banana_demo(args.v, args.omega, args.dt, [0.02, 0.002, 0.002, 0.02, 0.0001, 0.0001], args.n,
                    save_path=str(Path(__file__).parent / 'banana_low_noise.png'))

    # Default noise
    print("\n--- Default noise ---")
    run_banana_demo(args.v, args.omega, args.dt, DEFAULT_ALPHAS, args.n,
                    save_path=str(Path(__file__).parent / 'banana_default.png'))

    # High noise
    print("\n--- High noise ---")
    run_banana_demo(args.v, args.omega, args.dt, [0.4, 0.04, 0.04, 0.4, 0.01, 0.01], args.n,
                    save_path=str(Path(__file__).parent / 'banana_high_noise.png'))
