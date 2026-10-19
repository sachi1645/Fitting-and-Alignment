
#!/usr/bin/env python3
"""Line-first RANSAC and circle fitting for EN3160 Assignment 2, Question 2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


@dataclass
class LineModel:
    normal: np.ndarray
    d: float
    sample_indices: tuple[int, int]
    inliers: np.ndarray
    sample_normal: np.ndarray
    sample_d: float
    sample_points: np.ndarray


@dataclass
class CircleModel:
    center: np.ndarray
    radius: float
    sample_indices: tuple[int, int, int]
    inliers: np.ndarray
    sample_center: np.ndarray
    sample_radius: float
    sample_points: np.ndarray


# ---------------------------------------------------------
# 1. Generate the point set specified in the assignment
# ---------------------------------------------------------

def generate_point_set(seed: int = 0, n_total: int = 100):
    """Generate 50 noisy circle points and 50 noisy line points."""

    if n_total % 2:
        raise ValueError("n_total must be even")

    rng = np.random.RandomState(seed)
    half_n = n_total // 2

    # Ground-truth circle: centre (2, 3), radius 10.
    r_gt, x0_gt, y0_gt = 10.0, 2.0, 3.0
    sigma_r = r_gt / 16.0

    t = rng.uniform(0.0, 2.0 * np.pi, half_n)
    noise = sigma_r * rng.randn(half_n)

    xc = x0_gt + (r_gt + noise) * np.cos(t)
    yc = y0_gt + (r_gt + noise) * np.sin(t)

    X_circle = np.column_stack((xc, yc))

    # Ground-truth line: y = -x + 2, with Gaussian noise.
    sigma_line, m_gt, b_gt = 1.0, -1.0, 2.0

    xl = np.linspace(-12.0, 12.0, half_n)
    yl = m_gt * xl + b_gt + sigma_line * rng.randn(half_n)

    X_line = np.column_stack((xl, yl))

    X = np.vstack((X_circle, X_line))

    labels = np.array(
        ["circle"] * half_n + ["line"] * half_n
    )

    # Unit normal representation: ax + by = d.
    normal_gt = np.array([1.0, 1.0]) / np.sqrt(2.0)
    d_gt = 2.0 / np.sqrt(2.0)

    ground_truth = {
        "center": np.array([x0_gt, y0_gt]),
        "radius": r_gt,
        "line_normal": normal_gt,
        "line_d": d_gt,
        "line_m": m_gt,
        "line_b": b_gt,
        "X_circle": X_circle,
        "X_line": X_line,
    }

    return X, labels, ground_truth


# ---------------------------------------------------------
# 2. Line estimation from two sample points
# ---------------------------------------------------------

def line_from_two_points(p1: np.ndarray, p2: np.ndarray):
    """Return unit normal n and d for n dot p = d."""

    delta = p2 - p1
    length = float(np.linalg.norm(delta))

    if length < 1e-10:
        return None

    tangent = delta / length

    # A perpendicular unit normal.
    normal = np.array(
        [-tangent[1], tangent[0]],
        dtype=float
    )

    normal /= np.linalg.norm(normal)

    d = float(np.dot(normal, p1))

    return normal, d


def line_residuals(
    points: np.ndarray,
    normal: np.ndarray,
    d: float
):
    """Perpendicular distances from points to a line."""

    # Correct because the normal has unit length.
    return np.abs(points @ normal - d)


def tls_line(points: np.ndarray):
    """Refine a line using total least squares."""

    center = points.mean(axis=0)

    _, _, vt = np.linalg.svd(
        points - center,
        full_matrices=False
    )

    direction = vt[0]
    direction /= np.linalg.norm(direction)

    normal = np.array(
        [-direction[1], direction[0]]
    )

    normal /= np.linalg.norm(normal)

    d = float(np.dot(normal, center))

    return normal, d


def ransac_line(
    points: np.ndarray,
    iterations: int = 2500,
    threshold: float = 1.7,
    min_inliers: int = 35,
    seed: int = 123
) -> LineModel:
    """Estimate a line using a custom RANSAC implementation."""

    if len(points) < 2:
        raise ValueError(
            "At least two points are required for line RANSAC"
        )

    rng = np.random.default_rng(seed)

    best_indices = None
    best_key = (-1, float("-inf"))

    for _ in range(iterations):

        # Randomly select two points.
        ids = tuple(
            int(i)
            for i in rng.choice(
                len(points),
                size=2,
                replace=False
            )
        )

        fit = line_from_two_points(
            points[ids[0]],
            points[ids[1]]
        )

        if fit is None:
            continue

        normal, d = fit

        residuals = line_residuals(
            points, normal, d
        )

        inliers = residuals <= threshold
        count = int(inliers.sum())

        # Prefer more inliers, then smaller residual error.
        error = float(
            np.sum(
                np.minimum(
                    residuals[inliers] ** 2,
                    threshold ** 2
                )
            )
        ) if count else float("inf")

        key = (count, -error)

        if key > best_key:
            best_key = key
            best_indices = ids
            best_inliers = inliers.copy()

            sample_normal = normal.copy()
            sample_d = d

    if (
        best_indices is None
        or int(best_inliers.sum()) < min_inliers
    ):
        count = (
            0 if best_indices is None
            else int(best_inliers.sum())
        )

        raise RuntimeError(
            f"Line RANSAC failed: {count} inliers; "
            f"minimum required is {min_inliers}."
        )

    # Refine the winning model using all its consensus points.
    normal, d = tls_line(points[best_inliers])

    final_inliers = (
        line_residuals(points, normal, d) <= threshold
    )

    return LineModel(
        normal=normal,
        d=d,
        sample_indices=best_indices,
        inliers=final_inliers,
        sample_normal=sample_normal,
        sample_d=sample_d,
        sample_points=points[list(best_indices)].copy()
    )


# ---------------------------------------------------------
# 3. Circle estimation from three sample points
# ---------------------------------------------------------

def circle_from_three_points(points: np.ndarray):
    """Calculate circle centre and radius from three points."""

    p1, p2, p3 = points

    matrix = 2.0 * np.array(
        [p2 - p1, p3 - p1],
        dtype=float
    )

    rhs = np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1)
    ])

    # Collinear or nearly collinear sample points.
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        return None

    center = np.linalg.solve(matrix, rhs)

    radius = float(
        np.linalg.norm(p1 - center)
    )

    if not np.isfinite(radius) or radius <= 0:
        return None

    return center, radius


def circle_residuals(
    points: np.ndarray,
    center: np.ndarray,
    radius: float
):
    """Absolute radial errors for a proposed circle."""

    distances = np.linalg.norm(
        points - center[None, :],
        axis=1
    )

    return np.abs(distances - radius)


def optimize_circle(
    points: np.ndarray,
    initial_center: np.ndarray,
    initial_radius: float
):
    """Minimize squared geometric radial residuals (not an algebraic circle fit)."""

    def objective(params):

        cx, cy, radius = params

        if radius <= 0:
            return 1e12 + 1e8 * abs(radius)

        center = np.array([cx, cy])

        residuals = (
            np.linalg.norm(
                points - center[None, :],
                axis=1
            ) - radius
        )

        return float(np.dot(residuals, residuals))

    start = np.array([
        initial_center[0],
        initial_center[1],
        max(1e-3, initial_radius)
    ], dtype=float)

    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={
            "maxiter": 5000,
            "xatol": 1e-8,
            "fatol": 1e-8
        }
    )

    if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"Circle optimization failed: {result.message}"
        )

    cx, cy, radius = result.x

    return np.array([cx, cy]), float(abs(radius))


def ransac_circle(
    points: np.ndarray,
    iterations: int = 4500,
    threshold: float = 1.25,
    min_inliers: int = 32,
    seed: int = 456
) -> CircleModel:
    """Estimate a circle using a custom RANSAC implementation."""

    if len(points) < 3:
        raise ValueError(
            "At least three points are required for circle RANSAC"
        )

    rng = np.random.default_rng(seed)

    best_indices = None
    best_key = (-1, float("-inf"))

    for _ in range(iterations):

        # Randomly choose three points.
        ids = tuple(
            int(i)
            for i in rng.choice(
                len(points),
                size=3,
                replace=False
            )
        )

        fit = circle_from_three_points(
            points[list(ids)]
        )

        if fit is None:
            continue

        center, radius = fit

        # Reject degenerate or excessively large circles.
        if radius > 100.0 or radius < 1.0:
            continue

        residuals = circle_residuals(
            points, center, radius
        )

        inliers = residuals <= threshold
        count = int(inliers.sum())

        error = float(
            np.sum(
                np.minimum(
                    residuals[inliers] ** 2,
                    threshold ** 2
                )
            )
        ) if count else float("inf")

        key = (count, -error)

        if key > best_key:
            best_key = key
            best_indices = ids
            best_inliers = inliers.copy()

            sample_center = center.copy()
            sample_radius = radius

    if (
        best_indices is None
        or int(best_inliers.sum()) < min_inliers
    ):
        count = (
            0 if best_indices is None
            else int(best_inliers.sum())
        )

        raise RuntimeError(
            f"Circle RANSAC failed: {count} inliers; "
            f"minimum required is {min_inliers}."
        )

    # Optimize using the consensus of the best sample.
    center, radius = optimize_circle(
        points[best_inliers],
        sample_center,
        sample_radius
    )

    final_inliers = (
        circle_residuals(
            points, center, radius
        ) <= threshold
    )

    # Refine once more using the updated consensus.
    if int(final_inliers.sum()) >= 3:

        center, radius = optimize_circle(
            points[final_inliers],
            center,
            radius
        )

        final_inliers = (
            circle_residuals(
                points, center, radius
            ) <= threshold
        )

    return CircleModel(
        center=center,
        radius=radius,
        sample_indices=best_indices,
        inliers=final_inliers,
        sample_center=sample_center,
        sample_radius=sample_radius,
        sample_points=points[list(best_indices)].copy()
    )


# ---------------------------------------------------------
# 4. Utility for plotting a line in normal form
# ---------------------------------------------------------

def line_y(normal, d, x):

    nx, ny = normal

    if abs(ny) < 1e-8:
        return np.full_like(x, np.nan, dtype=float)

    return (d - nx * x) / ny


# ---------------------------------------------------------
# 5. Main pipeline: line first, then circle
# ---------------------------------------------------------

def run(
    seed: int = 0,
    output_dir: str = "results/q2",
    show: bool = False
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, labels, gt = generate_point_set(seed=seed)

    # Part (a): detect the line using RANSAC.
    line = ransac_line(
        X,
        threshold=1.7,
        min_inliers=35,
        seed=seed + 1
    )

    # Remove the line consensus before circle fitting.
    remainder_indices = np.flatnonzero(~line.inliers)
    X_remaining = X[remainder_indices]

    # Part (b): fit a circle to the remaining points.
    circle = ransac_circle(
        X_remaining,
        threshold=1.25,
        min_inliers=32,
        seed=seed + 2
    )

    # Map circle inliers back to the original 100-point set.
    circle_global_inliers = np.zeros(
        len(X), dtype=bool
    )

    circle_global_inliers[
        remainder_indices[circle.inliers]
    ] = True

    # Part (d): reverse the order for comparison.
    circle_first = ransac_circle(
        X,
        threshold=1.25,
        min_inliers=32,
        seed=seed + 2
    )

    circle_first_remaining_indices = np.flatnonzero(
        ~circle_first.inliers
    )

    circle_first_line = ransac_line(
        X[circle_first_remaining_indices],
        threshold=1.7,
        min_inliers=35,
        seed=seed + 1
    )

    # -----------------------------------------------------
    # Part (c): plot all points, samples, inliers and models
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8.5, 7.8))

    # All points.
    ax.scatter(
        X[:, 0], X[:, 1],
        s=20,
        c="#9aa0a6",
        alpha=0.6,
        label="All points",
        zorder=1
    )

    # Inliers for the line.
    ax.scatter(
        X[line.inliers, 0],
        X[line.inliers, 1],
        s=26,
        c="#e0a800",
        label=f"Line inliers ({line.inliers.sum()})",
        zorder=3
    )

    # Inliers for the circle.
    ax.scatter(
        X[circle_global_inliers, 0],
        X[circle_global_inliers, 1],
        s=28,
        c="#10aeb4",
        label=(
            f"Circle inliers "
            f"({circle_global_inliers.sum()})"
        ),
        zorder=4
    )

    # Two points from the best line sample.
    ax.scatter(
        line.sample_points[:, 0],
        line.sample_points[:, 1],
        s=95,
        marker="x",
        c="#d62728",
        linewidths=2.2,
        label="Best line sample (2 points)",
        zorder=7
    )

    # Three points from the best circle sample.
    ax.scatter(
        circle.sample_points[:, 0],
        circle.sample_points[:, 1],
        s=95,
        marker="*",
        c="#7b1fa2",
        edgecolors="white",
        linewidths=0.5,
        label="Best circle sample (3 points)",
        zorder=8
    )

    xgrid = np.linspace(
        X[:, 0].min() - 1,
        X[:, 0].max() + 1,
        350
    )

    # Ground-truth line.
    ax.plot(
        xgrid,
        line_y(
            gt["line_normal"],
            gt["line_d"],
            xgrid
        ),
        "--",
        color="#6c757d",
        lw=1.5,
        label="Ground-truth line"
    )

    # Line from the two points that produced the best sample.
    ax.plot(
        xgrid,
        line_y(
            line.sample_normal,
            line.sample_d,
            xgrid
        ),
        color="#e63946",
        lw=1.1,
        alpha=0.85,
        label="RANSAC sample line"
    )

    # Refined best-fit line.
    ax.plot(
        xgrid,
        line_y(line.normal, line.d, xgrid),
        color="#f08a24",
        lw=2.0,
        label="Best-fit TLS line"
    )

    angles = np.linspace(
        0, 2 * np.pi, 500
    )

    # Ground-truth circle.
    ax.plot(
        gt["center"][0] + gt["radius"] * np.cos(angles),
        gt["center"][1] + gt["radius"] * np.sin(angles),
        color="#7cb342",
        ls="--",
        lw=1.4,
        label="Ground-truth circle"
    )

    # Circle estimated only from the winning three-point sample.
    ax.plot(
        circle.sample_center[0]
        + circle.sample_radius * np.cos(angles),
        circle.sample_center[1]
        + circle.sample_radius * np.sin(angles),
        color="#7b1fa2",
        lw=1.1,
        alpha=0.8,
        label="Circle from best sample"
    )

    # Optimized best-fit circle.
    ax.plot(
        circle.center[0] + circle.radius * np.cos(angles),
        circle.center[1] + circle.radius * np.sin(angles),
        color="#006d77",
        lw=2.0,
        label="Optimized best-fit circle"
    )

    ax.set(
        xlabel="x",
        ylabel="y",
        title="RANSAC line-first fitting: line and circle"
    )

    ax.grid(True, alpha=0.25)
    ax.axis("equal")

    ax.legend(
        fontsize=7.4,
        loc="upper right",
        framealpha=0.92
    )

    fig.tight_layout()

    plot_path = out / "ransac_line_circle.png"

    fig.savefig(
        plot_path,
        dpi=180,
        bbox_inches="tight"
    )

    if show:
        plt.show()

    plt.close(fig)

    # -----------------------------------------------------
    # 6. Save numerical results for the report
    # -----------------------------------------------------

    fitted_slope = (
        -line.normal[0] / line.normal[1]
        if abs(line.normal[1]) > 1e-9
        else float("inf")
    )

    fitted_intercept = (
        line.d / line.normal[1]
        if abs(line.normal[1]) > 1e-9
        else float("nan")
    )

    metrics_path = out / "metrics.txt"

    with metrics_path.open(
        "w", encoding="utf-8"
    ) as f:

        f.write(f"Random seed: {seed}\n")

        f.write(
            f"Line inliers: {int(line.inliers.sum())}/100; "
            "threshold=1.7; min consensus=35\n"
        )

        f.write(
            f"Line unit normal: "
            f"[{line.normal[0]:.6f}, "
            f"{line.normal[1]:.6f}], "
            f"norm={np.linalg.norm(line.normal):.6f}\n"
        )

        f.write(
            f"Line d: {line.d:.6f}; "
            f"approximate y=m*x+b: "
            f"m={fitted_slope:.6f}, "
            f"b={fitted_intercept:.6f}\n"
        )

        f.write(
            f"Line sample points: "
            f"{line.sample_points.tolist()}\n"
        )

        f.write(
            f"Circle inliers after line removal: "
            f"{int(circle_global_inliers.sum())}; "
            "threshold=1.25; min consensus=32\n"
        )

        f.write(
            f"Circle center: "
            f"({circle.center[0]:.6f}, "
            f"{circle.center[1]:.6f}); "
            f"radius={circle.radius:.6f}\n"
        )

        f.write(
            f"Circle sample points: "
            f"{circle.sample_points.tolist()}\n"
        )

        # Check the true composition of the consensuses.
        line_true = int(
            np.sum(
                line.inliers & (labels == "line")
            )
        )

        line_circle = int(
            np.sum(
                line.inliers & (labels == "circle")
            )
        )

        circle_true = int(
            np.sum(
                circle_global_inliers
                & (labels == "circle")
            )
        )

        circle_line = int(
            np.sum(
                circle_global_inliers
                & (labels == "line")
            )
        )

        f.write(
            f"Line consensus composition: "
            f"{line_true} true-line + "
            f"{line_circle} true-circle points\n"
        )

        f.write(
            f"Circle consensus composition: "
            f"{circle_true} true-circle + "
            f"{circle_line} true-line points\n"
        )

        # Results when circle fitting is attempted first.
        cf_circle_true = int(
            np.sum(
                circle_first.inliers
                & (labels == "circle")
            )
        )

        cf_circle_line = int(
            np.sum(
                circle_first.inliers
                & (labels == "line")
            )
        )

        remaining_labels = labels[
            circle_first_remaining_indices
        ]

        cf_line_true = int(
            np.sum(
                circle_first_line.inliers
                & (remaining_labels == "line")
            )
        )

        cf_line_circle = int(
            np.sum(
                circle_first_line.inliers
                & (remaining_labels == "circle")
            )
        )

        f.write(
            f"Circle-first: "
            f"{int(circle_first.inliers.sum())} "
            f"circle consensus = "
            f"{cf_circle_true} true-circle + "
            f"{cf_circle_line} true-line points\n"
        )

        f.write(
            f"Circle-first remainder: "
            f"{len(circle_first_remaining_indices)} points; "
            f"line fit inliers="
            f"{int(circle_first_line.inliers.sum())} "
            f"({cf_line_true} true-line + "
            f"{cf_line_circle} true-circle)\n"
        )

        f.write(
            "Ground truth line: y=-x+2; "
            "circle center=(2,3), radius=10\n"
        )

    print(metrics_path.read_text())

    print(f"Plot saved: {plot_path.resolve()}")

    return {
        "X": X,
        "line": line,
        "circle": circle,
        "circle_global_inliers": circle_global_inliers,
        "ground_truth": gt,
        "circle_first": circle_first,
        "circle_first_line": circle_first_line,
    }


# ---------------------------------------------------------
# 7. Program entry point
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    parser.add_argument(
        "--output-dir",
        default="results/q2"
    )

    parser.add_argument(
        "--show",
        action="store_true"
    )

    args = parser.parse_args()

    run(
        seed=args.seed,
        output_dir=args.output_dir,
        show=args.show
    )


if __name__ == "__main__":
    main()
