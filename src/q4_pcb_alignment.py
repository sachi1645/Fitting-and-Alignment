
#!/usr/bin/env python3
"""SIFT + BFMatcher + RANSAC homography + aligned-image differencing (Q4)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def align_and_compare(
    template_path: Path,
    pcb_path: Path,
    output_dir: Path,
    ratio: float = 0.75,
    ransac_threshold: float = 5.0,
):
    """Align one PCB image to the template and compute differences."""

    template = cv2.imread(
        str(template_path), cv2.IMREAD_COLOR
    )

    pcb = cv2.imread(
        str(pcb_path), cv2.IMREAD_COLOR
    )

    if template is None:
        raise FileNotFoundError(
            f"Cannot read template: {template_path}"
        )

    if pcb is None:
        raise FileNotFoundError(
            f"Cannot read PCB image: {pcb_path}"
        )

    # -----------------------------------------------------
    # Step 1: Detect SIFT features in both images.
    # -----------------------------------------------------

    sift = cv2.SIFT_create()

    template_gray = cv2.cvtColor(
        template, cv2.COLOR_BGR2GRAY
    )

    pcb_gray = cv2.cvtColor(
        pcb, cv2.COLOR_BGR2GRAY
    )

    key_template, desc_template = sift.detectAndCompute(
        template_gray, None
    )

    key_pcb, desc_pcb = sift.detectAndCompute(
        pcb_gray, None
    )

    if (
        desc_template is None
        or desc_pcb is None
        or len(key_template) < 4
        or len(key_pcb) < 4
    ):
        raise RuntimeError(
            f"Not enough SIFT features in "
            f"{pcb_path.name} or template"
        )

    # -----------------------------------------------------
    # Step 2: Match SIFT descriptors using BFMatcher.
    # SIFT descriptors are floating-point vectors;
    # use the L2 distance metric.
    # -----------------------------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    pairs = matcher.knnMatch(
        desc_pcb,
        desc_template,
        k=2,
    )

    # Lowe's ratio test removes ambiguous feature matches.
    good = [
        m for m, n in pairs
        if m.distance < ratio * n.distance
    ]

    if len(good) < 4:
        raise RuntimeError(
            f"Only {len(good)} ratio-test matches "
            f"for {pcb_path.name}; need at least 4"
        )

    # Source points belong to the PCB.
    src_pts = np.float32([
        key_pcb[m.queryIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    # Destination points belong to the template.
    dst_pts = np.float32([
        key_template[m.trainIdx].pt
        for m in good
    ]).reshape(-1, 1, 2)

    # -----------------------------------------------------
    # Step 3: Estimate the homography with RANSAC.
    # -----------------------------------------------------

    H, inlier_mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransac_threshold,
    )

    if H is None or inlier_mask is None:
        raise RuntimeError(
            f"RANSAC homography estimation failed "
            f"for {pcb_path.name}"
        )

    inliers = inlier_mask.ravel().astype(bool)

    inlier_matches = [
        m for m, keep in zip(good, inliers)
        if keep
    ]

    # -----------------------------------------------------
    # Step 4: Warp the PCB into template coordinates.
    # -----------------------------------------------------

    th, tw = template.shape[:2]

    aligned = cv2.warpPerspective(
        pcb,
        H,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Warp a validity mask as well. This identifies which
    # output pixels actually come from the input PCB.
    source_valid = np.full(
        pcb.shape[:2],
        255,
        dtype=np.uint8,
    )

    warped_valid = cv2.warpPerspective(
        source_valid,
        H,
        (tw, th),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    valid = warped_valid > 0

    # -----------------------------------------------------
    # Step 5: Compute the absolute difference image.
    # -----------------------------------------------------

    diff = cv2.absdiff(
        template,
        aligned,
    )

    diff_gray = cv2.cvtColor(
        diff,
        cv2.COLOR_BGR2GRAY,
    )

    # Exclude uncovered pixels from the difference analysis.
    diff_gray[~valid] = 0

    # A colour heatmap makes larger differences easier to see.
    heat = cv2.applyColorMap(
        diff_gray,
        cv2.COLORMAP_JET,
    )

    heat[~valid] = (0, 0, 0)

    # Blend the template and aligned PCB for visual comparison.
    overlay = cv2.addWeighted(
        template, 0.55,
        aligned, 0.45,
        0,
    )

    # Show the original template in uncovered regions.
    overlay[~valid] = template[~valid]

    # -----------------------------------------------------
    # Step 6: Save all outputs.
    # -----------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = pcb_path.stem

    # Limit the displayed match lines for readability.
    # The metrics still use every ratio-test match/inlier.
    preview_matches = sorted(
        inlier_matches,
        key=lambda m: m.distance,
    )[:75]

    matches_vis = cv2.drawMatches(
        pcb,
        key_pcb,
        template,
        key_template,
        preview_matches,
        None,
        matchColor=(30, 210, 70),
        singlePointColor=(240, 180, 30),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    outputs = {
        "matches": output_dir / f"{stem}_matches.jpg",
        "aligned": output_dir / f"{stem}_aligned.jpg",
        "difference_gray": (
            output_dir / f"{stem}_difference_gray.png"
        ),
        "difference_heatmap": (
            output_dir / f"{stem}_difference_heatmap.png"
        ),
        "overlay": (
            output_dir / f"{stem}_overlay_comparison.jpg"
        ),
        "homography": (
            output_dir / f"{stem}_homography.txt"
        ),
    }

    if not cv2.imwrite(
        str(outputs["matches"]), matches_vis
    ):
        raise OSError(
            f"Could not save {outputs['matches']}"
        )

    if not cv2.imwrite(
        str(outputs["aligned"]), aligned
    ):
        raise OSError(
            f"Could not save {outputs['aligned']}"
        )

    if not cv2.imwrite(
        str(outputs["difference_gray"]), diff_gray
    ):
        raise OSError(
            f"Could not save {outputs['difference_gray']}"
        )

    if not cv2.imwrite(
        str(outputs["difference_heatmap"]), heat
    ):
        raise OSError(
            f"Could not save {outputs['difference_heatmap']}"
        )

    if not cv2.imwrite(
        str(outputs["overlay"]), overlay
    ):
        raise OSError(
            f"Could not save {outputs['overlay']}"
        )

    np.savetxt(
        outputs["homography"],
        H,
        fmt="%.8f",
    )

    # -----------------------------------------------------
    # Step 7: Quantify differences over valid pixels only.
    # -----------------------------------------------------

    valid_count = int(valid.sum())

    if valid_count:
        mean_diff = float(
            diff_gray[valid].mean()
        )

        median_diff = float(
            np.median(diff_gray[valid])
        )

        p95_diff = float(
            np.percentile(
                diff_gray[valid],
                95,
            )
        )
    else:
        mean_diff = float("nan")
        median_diff = float("nan")
        p95_diff = float("nan")

    metrics = {
        "pcb_image": pcb_path.name,
        "template": template_path.name,
        "template_keypoints": len(key_template),
        "pcb_keypoints": len(key_pcb),
        "ratio_test_matches": len(good),
        "ransac_inliers": int(inliers.sum()),
        "inlier_ratio": (
            float(inliers.mean())
            if len(inliers)
            else 0.0
        ),
        "valid_overlap_pixels": valid_count,
        "mean_abs_gray_difference": mean_diff,
        "median_abs_gray_difference": median_diff,
        "p95_abs_gray_difference": p95_diff,
    }

    print(metrics)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--template",
        default="images/pcb/AECC1.jpg",
        help="Reference PCB template image",
    )

    parser.add_argument(
        "--input-dir",
        default="images/pcb",
        help="Directory containing PCB images",
    )

    parser.add_argument(
        "--output-dir",
        default="results/q4",
        help="Directory for alignment and difference outputs",
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio-test threshold",
    )

    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold in pixels",
    )

    args = parser.parse_args()

    template_path = Path(args.template)
    input_dir = Path(args.input_dir)

    if not template_path.exists():
        raise SystemExit(
            f"Missing required course template: {template_path}\n"
            "Add AECC1.jpg to images/pcb/ or pass --template."
        )

    if not input_dir.is_dir():
        raise SystemExit(
            f"Missing PCB input directory: {input_dir}"
        )

    # Process every image in the input directory except
    # the reference template itself.
    files = sorted(
        pth
        for pth in input_dir.iterdir()
        if (
            pth.is_file()
            and pth.suffix.lower() in IMAGE_EXTS
            and pth.resolve() != template_path.resolve()
        )
    )

    if not files:
        raise SystemExit(
            f"No PCB test images found in {input_dir}; "
            "only the template may be present."
        )

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for path in files:
        try:
            result = align_and_compare(
                template_path,
                path,
                out,
                args.ratio,
                args.ransac_threshold,
            )

            results.append(result)

        except Exception as exc:
            print(
                f"ERROR for {path.name}: {exc}"
            )

            results.append({
                "pcb_image": path.name,
                "error": str(exc),
            })

    # Save the summary metrics for all test images.
    keys = sorted({
        key
        for row in results
        for key in row.keys()
    })

    summary_path = out / "summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=keys,
        )

        writer.writeheader()
        writer.writerows(results)

    successful = sum(
        "error" not in row
        for row in results
    )

    print(
        f"Processed {successful}/{len(files)} PCB images."
    )

    print(
        f"Results saved to: {out.resolve()}"
    )

    print(
        f"Summary CSV: {summary_path.resolve()}"
    )


if __name__ == "__main__":
    main()
