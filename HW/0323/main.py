import numpy as np
import cv2
from pathlib import Path

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
    return None


def proc(path):
    org_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if org_img is not None:
        pre_proc(org_img.copy())
    return None


if __name__ == "__main__":
    inp = Path("input/")
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
