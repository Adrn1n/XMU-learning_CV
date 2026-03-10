import numpy as np
import cv2
from pathlib import Path

GUAS_BLUR_KERNEL_SIZES = (5, 5)
GUAS_BLUR_SIGMA = 1
MED_BLUR_KERNEL_SIZES = 13
IS_TAR_BLACK = True
DRAW_COUNTER_COLOR = (0, 0, 255)
DRAW_COUNTER_THICKNESS = 2
COUNTER_FONT_SCALE = 0.5
COUNTER_FONT_COLOR = (0, 255, 0)
COUNTER_FONT_THICKNESS = 2
CNTR_FONT_SCALE = 1
CNTR_FONT_COLOR = (255, 0, 0)
CNTR_FONT_THICKNESS = 2
EXTS = ("*.jpg",)


def cvrt2uint8(img):
    if img.dtype == np.uint8:
        return img
    return cv2.convertScaleAbs(
        cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX)
    )


def pre_proc(img):
    if img is not None:
        img = cvrt2uint8(img)
        return cv2.medianBlur(
            cv2.GaussianBlur(
                cv2.equalizeHist(
                    (
                        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if img.shape[2] == 3
                        else cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                    )
                    if img.ndim > 2
                    else img
                ),
                GUAS_BLUR_KERNEL_SIZES,
                sigmaX=GUAS_BLUR_SIGMA,
                sigmaY=GUAS_BLUR_SIGMA,
            ),
            MED_BLUR_KERNEL_SIZES,
        )
    return None


def proc(path):
    org_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if org_img is not None:
        img = pre_proc(org_img.copy())
        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if org_img.ndim == 2:
            org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2BGR)
        if IS_TAR_BLACK:
            bin_img = cv2.bitwise_not(bin_img)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = 0
        for contour in contours:
            cv2.drawContours(
                org_img, [contour], -1, DRAW_COUNTER_COLOR, DRAW_COUNTER_THICKNESS
            )
            cnt += 1
            m = cv2.moments(contour)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
            else:
                cx, cy = -1, -1
            cv2.putText(
                org_img,
                str(cnt),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                COUNTER_FONT_SCALE,
                COUNTER_FONT_COLOR,
                COUNTER_FONT_THICKNESS,
            )
        s = str(cnt)
        (_, font_height), _ = cv2.getTextSize(
            s, cv2.FONT_HERSHEY_SIMPLEX, CNTR_FONT_SCALE, CNTR_FONT_THICKNESS
        )
        cv2.putText(
            org_img,
            f"Count: {cnt}",
            (0, font_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            CNTR_FONT_SCALE,
            CNTR_FONT_COLOR,
            CNTR_FONT_THICKNESS,
        )
        return org_img
    return None


if __name__ == "__main__":
    inp = Path("dataset/")
    out = Path("output/")
    out.mkdir(exist_ok=True)
    files = []
    for e in EXTS:
        files.extend(inp.glob(e))
    if files:
        for f in files:
            name, ext = f.stem, f.suffix
            res = proc(f)
            if res is not None:
                cv2.imwrite(str(out / f"{name}{ext}"), res)
            else:
                print(f"Failed to process {f}")
    else:
        print(f"No image files found in {inp.resolve()}")
