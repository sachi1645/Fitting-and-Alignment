
#!/usr/bin/env python3
"""Perspective placement of a flag/poster onto a planar image surface (Q3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def composite(
    background: np.ndarray,
    overlay: np.ndarray,
    dst_points,
    alpha: float = 1.0,
):
    """Warp an overlay rectangle to four destination points and blend it in."""

    if background is None or overlay is None:
        raise ValueError(
            "Background and overlay images must both be readable"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    h, w = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # Four source-image corners in this order:
    # top-left, top-right, bottom-right, bottom-left.
    src = np.array(
        [
            [0, 0],
            [ow - 1, 0],
            [ow - 1, oh - 1],
            [0, oh - 1],
        ],
        dtype=np.float32,
    )

    # Corresponding four corners on the background plane.
    dst = np.asarray(
        dst_points, dtype=np.float32
    ).reshape(4, 2)

    # Estimate the projective transformation.
    H = cv2.getPerspectiveTransform(src, dst)

    # Warp the overlay into the background image coordinates.
    warped = cv2.warpPerspective(
        overlay,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Warp a mask so pixels outside the overlay are not blended.
    source_mask = np.full(
        (oh, ow), 255, dtype=np.uint8
    )

    warped_mask = cv2.warpPerspective(
        source_mask,
        H,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mask = warped_mask > 0

    # Blend only inside the projected quadrilateral.
    out = background.copy()

    out[mask] = np.clip(
        alpha * warped[mask]
        + (1.0 - alpha) * background[mask],
        0,
        255,
    ).astype(np.uint8)

    # Draw a thin red outline for visualising the target plane.
    cv2.polylines(
        out,
        [dst.astype(np.int32)],
        isClosed=True,
        color=(0, 0, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return out, H


def choose_destination_points(image: np.ndarray):
    """
    Interactively select four corners in the order:
    TL -> TR -> BR -> BL.
    """

    points = []
    display = image.copy()

    title = (
        "Click plane corners: TL, TR, BR, BL; "
        "press ESC to cancel"
    )

    def callback(event, x, y, flags, userdata):

        if (
            event == cv2.EVENT_LBUTTONDOWN
            and len(points) < 4
        ):
            points.append((x, y))

            cv2.circle(
                display,
                (x, y),
                5,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                display,
                str(len(points)),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.imshow(title, display)

    cv2.imshow(title, display)
    cv2.setMouseCallback(title, callback)

    try:
        while len(points) < 4:
            key = cv2.waitKey(25) & 0xFF

            if key == 27:
                raise SystemExit(
                    "Point selection cancelled"
                )
    finally:
        cv2.destroyAllWindows()

    return points


def process_example(root: Path, item: dict):
    """Process one background/overlay image pair."""

    bg_path = root / item["background"]
    ov_path = root / item["overlay"]

    background = cv2.imread(
        str(bg_path), cv2.IMREAD_COLOR
    )

    overlay = cv2.imread(
        str(ov_path), cv2.IMREAD_COLOR
    )

    if background is None:
        raise FileNotFoundError(
            f"Cannot read background image: {bg_path}"
        )

    if overlay is None:
        raise FileNotFoundError(
            f"Cannot read overlay image: {ov_path}"
        )

    # Use configured points or ask the user to select four.
    dst = item.get("destination_points")

    if dst is None:
        dst = choose_destination_points(background)

    result, H = composite(
        background,
        overlay,
        dst,
        float(item.get("alpha", 1.0)),
    )

    output_path = root / item["output"]
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    if not cv2.imwrite(str(output_path), result):
        raise OSError(
            f"Failed to write output: {output_path}"
        )

    homography_path = output_path.with_suffix(
        ".homography.txt"
    )

    np.savetxt(
        homography_path,
        H,
        fmt="%.8f",
    )

    print(f"\nBackground: {bg_path.name}")
    print(f"Overlay: {ov_path.name}")
    print(f"Output: {output_path}")
    print(f"Homography saved to: {homography_path}")
    print("Homography matrix:")
    print(H)
    print("Destination points:", np.asarray(dst).tolist())


def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        default="q3_examples.json",
        help="JSON configuration file; paths are relative to project root",
    )

    parser.add_argument(
        "--interactive-background",
        help="Background image for interactive four-corner selection",
    )

    parser.add_argument(
        "--interactive-overlay",
        help="Overlay image for interactive placement",
    )

    parser.add_argument(
        "--output",
        default="results/q3/interactive_composite.png",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Overlay opacity between 0 and 1",
    )

    args = parser.parse_args()

    # The project root is the parent directory of src/.
    root = Path(__file__).resolve().parents[1]

    # Optional interactive mode for your own image pair.
    if args.interactive_background:

        bg_path = Path(args.interactive_background)
        ov_path = (
            Path(args.interactive_overlay)
            if args.interactive_overlay
            else None
        )

        if not bg_path.is_absolute():
            bg_path = root / bg_path

        if ov_path is not None and not ov_path.is_absolute():
            ov_path = root / ov_path

        if ov_path is None:
            raise SystemExit(
                "Please provide --interactive-overlay as well."
            )

        background = cv2.imread(
            str(bg_path), cv2.IMREAD_COLOR
        )

        overlay = cv2.imread(
            str(ov_path), cv2.IMREAD_COLOR
        )

        if background is None:
            raise SystemExit(
                f"Cannot read background: {bg_path}"
            )

        if overlay is None:
            raise SystemExit(
                f"Cannot read overlay: {ov_path}"
            )

        dst = choose_destination_points(background)

        result, H = composite(
            background,
            overlay,
            dst,
            args.alpha,
        )

        output_path = Path(args.output)

        if not output_path.is_absolute():
            output_path = root / output_path

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        if not cv2.imwrite(
            str(output_path), result
        ):
            raise OSError(
                f"Failed to write output: {output_path}"
            )

        np.savetxt(
            output_path.with_suffix(".homography.txt"),
            H,
            fmt="%.8f",
        )

        print(f"Saved composite: {output_path}")
        print("Homography matrix:")
        print(H)

        return

    # Standard mode: process all configured examples.
    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = root / config_path

    if not config_path.exists():
        raise SystemExit(
            f"Configuration file not found: {config_path}"
        )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    examples = config.get("examples", [])

    if not examples:
        raise SystemExit(
            "No examples were found in the configuration file."
        )

    for item in examples:
        process_example(root, item)


if __name__ == "__main__":
    main()
