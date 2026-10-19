
#!/usr/bin/env python3
"""Scale-normalized LoG blob detection for the EN3160 sunflower image."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter


def manual_log_kernel(sigma: float) -> np.ndarray:
    """Construct a finite, scale-normalized Laplacian-of-Gaussian kernel.

    Equivalent in concept to the hand-built kernels in the supplied reference
    notebook, but normalized and mean-corrected before convolution.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    half_width = max(2, int(np.ceil(3.0 * sigma)))
    yy, xx = np.mgrid[-half_width:half_width+1, -half_width:half_width+1]
    radius_sq = xx.astype(float)**2 + yy.astype(float)**2
    kernel = ((radius_sq / (2 * sigma**2) - 1) *
              np.exp(-radius_sq / (2 * sigma**2)) / (np.pi * sigma**4))
    kernel *= sigma**2
    # Remove the small numerical DC response introduced by truncation.
    kernel -= kernel.mean()
    return kernel.astype(np.float32)


def log_response(gray: np.ndarray, sigma: float, backend: str = "scipy") -> np.ndarray:
    """Absolute, scale-normalized LoG magnitude using either implementation."""
    if backend == "scipy":
        return np.abs(sigma**2 * gaussian_laplace(gray, sigma=float(sigma), mode="reflect"))
    if backend == "manual":
        kernel = manual_log_kernel(float(sigma))
        return np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101))
    raise ValueError("backend must be 'scipy' or 'manual'")


def detect_blobs(
    image_bgr: np.ndarray,
    sigma_min: float = 0.5,
    sigma_max: float = 8.0,
    num_scales: int = 32,
    threshold: float = 0.06,
    spatial_window: int = 7,
    roi_y_min: int = 180,
    backend: str = "scipy",
):
    """Detect scale-space LoG extrema.

    Returns:
        sigmas: scales tested.
        detections: (x, y, sigma, radius, response) tuples.
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")

    if sigma_min <= 0 or sigma_max <= sigma_min:
        raise ValueError("Require 0 < sigma_min < sigma_max")

    if num_scales < 3:
        raise ValueError("num_scales must be at least 3")

    if spatial_window < 3:
        raise ValueError("spatial_window must be >= 3")

    if spatial_window % 2 == 0:
        spatial_window += 1

    # Convert the image to grayscale and normalize pixel intensities.
    gray = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0

    # Examine small to large blobs using logarithmically spaced scales.
    sigmas = np.geomspace(
        sigma_min, sigma_max, num_scales
    )

    # Scale-normalized absolute LoG responses, including a hand-built kernel option.
    stack = np.stack([
        log_response(gray, float(sigma), backend=backend)
        for sigma in sigmas
    ], axis=0)

    # Find local maxima in both spatial and scale dimensions.
    local_max = maximum_filter(
        stack,
        size=(3, spatial_window, spatial_window),
        mode="nearest",
    )

    peaks = (
        (stack >= local_max - 1e-8)
        & (stack >= float(threshold))
    )

    # Ignore the first and last scales because they cannot be
    # compared with neighbouring scales on both sides.
    scale_ids, ys, xs = np.where(peaks[1:-1])

    h, w = gray.shape
    rows = []

    for s0, y, x in zip(
        scale_ids.tolist(),
        ys.tolist(),
        xs.tolist(),
    ):
        # Ignore the sky and upper background region.
        if y < int(roi_y_min):
            continue

        sigma = float(sigmas[s0 + 1])

        # Approximate blob radius for the LoG scale.
        radius = float(np.sqrt(2.0) * sigma)

        # Avoid detections whose estimated circles cross
        # the image boundary.
        if (
            x < radius
            or y < radius
            or x >= w - radius
            or y >= h - radius
        ):
            continue

        response = float(stack[s0 + 1, y, x])

        rows.append(
            (x, y, sigma, radius, response)
        )

    # Sort strongest candidates first.
    rows.sort(key=lambda row: row[4], reverse=True)

    # Non-maximum suppression across nearby scales:
    # remove duplicate detections of the same blob.
    accepted = []

    for row in rows:
        x, y, sigma, radius, response = row

        duplicate = any(
            np.hypot(x - ax, y - ay)
            < max(4.0, 0.65 * min(radius, aradius))
            and abs(np.log(sigma / asigma)) < 0.45
            for ax, ay, asigma, aradius, _ in accepted
        )

        if not duplicate:
            accepted.append(row)

    # Display and report larger circles first.
    accepted.sort(key=lambda row: row[3], reverse=True)

    return sigmas, accepted


def validate_sunflower_candidates(
    image_bgr: np.ndarray,
    detections,
    min_yellow_fraction: float = 0.28,
    min_ring_contrast: float = 8.0,
):
    """Reject some foliage and background detections.

    The LoG detector determines each candidate's centre and scale.
    This additional check looks for yellow petals around a darker
    centre, as commonly seen in this sunflower image.
    """

    hsv = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    # Yellow pixels in OpenCV's HSV colour space.
    yellow = (
        (hsv[:, :, 0] >= 12)
        & (hsv[:, :, 0] <= 36)
        & (hsv[:, :, 1] >= 80)
        & (hsv[:, :, 2] >= 70)
    )

    h, w = gray.shape
    kept = []

    for row in detections:
        x, y, sigma, radius, response = row

        # Extract a local patch around the candidate.
        x0 = max(0, int(x - 2 * radius))
        x1 = min(w, int(x + 2 * radius + 1))

        y0 = max(0, int(y - 2 * radius))
        y1 = min(h, int(y + 2 * radius + 1))

        yy, xx = np.ogrid[y0:y1, x0:x1]

        distance = np.sqrt(
            (xx - x) ** 2 + (yy - y) ** 2
        )

        # Centre and surrounding annulus.
        center = distance <= 0.4 * radius

        ring = (
            (distance >= 0.6 * radius)
            & (distance <= 1.4 * radius)
        )

        if not center.any() or not ring.any():
            continue

        yellow_patch = yellow[y0:y1, x0:x1]
        gray_patch = gray[y0:y1, x0:x1]

        yellow_fraction = float(
            yellow_patch[ring].mean()
        )

        # A sunflower centre is often darker than the petals.
        contrast = float(
            gray_patch[ring].mean()
            - gray_patch[center].mean()
        )

        if (
            yellow_fraction >= min_yellow_fraction
            and contrast >= min_ring_contrast
        ):
            kept.append(
                (row, yellow_fraction, contrast)
            )

    kept.sort(
        key=lambda item: item[0][3],
        reverse=True,
    )

    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--image",
        default="images/the_berry_farms_sunflower_field.jpeg",
    )

    parser.add_argument(
        "--output-dir",
        default="results/q1",
    )

    parser.add_argument(
        "--sigma-min",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--sigma-max",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--scales",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help=(
            "Draw this many largest detections; "
            "0 or less draws all validated detections"
        ),
    )

    parser.add_argument(
        "--spatial-window",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--roi-y-min",
        type=int,
        default=180,
        help=(
            "Ignore candidates above this row "
            "in the reduced image"
        ),
    )

    parser.add_argument(
        "--min-yellow-fraction",
        type=float,
        default=0.28,
    )

    parser.add_argument(
        "--min-ring-contrast",
        type=float,
        default=8.0,
    )

    parser.add_argument("--log-backend", choices=("scipy", "manual"), default="scipy",
                        help="Use SciPy LoG (default) or an explicitly constructed LoG kernel")

    args = parser.parse_args()

    # Load the supplied image.
    image_path = Path(args.image)

    if not image_path.exists():
        raise SystemExit(
            f"Missing required course image: {image_path}\n"
            "Place the original sunflower JPEG at this path "
            "or pass --image."
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_REDUCED_COLOR_4,
    )

    if image is None:
        raise SystemExit(
            f"OpenCV could not read {image_path}"
        )

    # Step 1: LoG scale-space detection.
    sigmas, raw_detections = detect_blobs(
        image,
        args.sigma_min,
        args.sigma_max,
        args.scales,
        args.threshold,
        spatial_window=args.spatial_window,
        roi_y_min=args.roi_y_min,
        backend=args.log_backend,
    )

    # Step 2: Reject some non-flower candidates.
    scored = validate_sunflower_candidates(
        image,
        raw_detections,
        args.min_yellow_fraction,
        args.min_ring_contrast,
    )

    detections = [
        item[0] for item in scored
    ]

    # Create the output directory.
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 3: Draw the detected circles.
    overlay = image.copy()

    draw_rows = (
        detections
        if args.top <= 0
        else detections[:args.top]
    )

    for x, y, sigma, radius, response in draw_rows:
        cv2.circle(
            overlay,
            (int(x), int(y)),
            max(1, int(round(radius))),
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # Mark the detected centre.
        cv2.circle(
            overlay,
            (int(x), int(y)),
            1,
            (255, 255, 0),
            -1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(out / "sunflower_blobs.jpg"),
        overlay,
    )

    # Step 4: Save all validated detections to CSV.
    csv_path = out / "detections.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "rank_by_radius",
            "x_px",
            "y_px",
            "sigma_px",
            "radius_px",
            "response",
            "yellow_annulus_fraction",
            "ring_minus_center_gray",
        ])

        for rank, (row, yellow_fraction, contrast) in enumerate(
            scored, start=1
        ):
            writer.writerow([
                rank,
                *[f"{value:.4f}" for value in row],
                f"{yellow_fraction:.4f}",
                f"{contrast:.3f}",
            ])

    # Step 5: Save parameters and the 15 largest detections.
    with (
        out / "parameters.txt"
    ).open("w", encoding="utf-8") as file:

        file.write(f"Image: {image_path}\n")
        file.write(f"LoG backend: {args.log_backend}\n")

        file.write(
            "OpenCV read mode: IMREAD_REDUCED_COLOR_4\n"
        )

        file.write(
            f"Reduced dimensions: "
            f"{image.shape[1]} x {image.shape[0]}\n"
        )

        file.write(
            f"Scale range: sigma in "
            f"[{sigmas[0]:.4f}, {sigmas[-1]:.4f}] "
            "reduced-image pixels\n"
        )

        file.write(
            f"Scales tested: {len(sigmas)} "
            "(logarithmically spaced)\n"
        )

        file.write(
            "Absolute scale-normalised LoG response "
            f"threshold: {args.threshold}\n"
        )

        file.write(
            "Spatial-scale maximum neighbourhood: "
            f"3 scales x {args.spatial_window} "
            f"x {args.spatial_window} pixels\n"
        )

        file.write(
            f"ROI: y >= {args.roi_y_min} pixels "
            "in the reduced image\n"
        )

        file.write(
            "Sunflower validation: yellow annulus "
            f"fraction >= {args.min_yellow_fraction}; "
            "ring-minus-centre grayscale >= "
            f"{args.min_ring_contrast}\n"
        )

        file.write(
            f"Raw LoG candidates after NMS: "
            f"{len(raw_detections)}\n"
        )

        file.write(
            f"Validated detections: "
            f"{len(detections)}\n"
        )

        file.write(
            "Largest circles "
            "(x, y, sigma, radius, response, "
            "yellow fraction, ring contrast):\n"
        )

        for rank, (row, yellow_fraction, contrast) in enumerate(
            scored[:15], start=1
        ):
            x, y, sigma, radius, response = row

            file.write(
                f"{rank:02d}: x={x}, y={y}, "
                f"sigma={sigma:.3f}, "
                f"radius={radius:.3f}, "
                f"response={response:.5f}, "
                f"yellow={yellow_fraction:.3f}, "
                f"contrast={contrast:.2f}\n"
            )

    # Print a summary in the terminal.
    print(
        f"Image dimensions after reduced read: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print(
        f"Sigma range tested: {sigmas[0]:.3f} .. "
        f"{sigmas[-1]:.3f} px ({len(sigmas)} scales)"
    )

    print(
        f"Raw LoG candidates: {len(raw_detections)}; "
        f"validated sunflower-like detections: "
        f"{len(detections)}"
    )

    print(
        "Largest detected circles (sorted by radius):"
    )

    for rank, (row, yellow_fraction, contrast) in enumerate(
        scored[:15], start=1
    ):
        x, y, sigma, radius, response = row

        print(
            f"{rank:02d}: center=({x},{y}), "
            f"sigma={sigma:.3f}, "
            f"radius={radius:.3f}, "
            f"response={response:.5f}, "
            f"yellow={yellow_fraction:.2f}, "
            f"contrast={contrast:.1f}"
        )

    print(
        f"Results saved to: {out.resolve()}"
    )


if __name__ == "__main__":
    main()
