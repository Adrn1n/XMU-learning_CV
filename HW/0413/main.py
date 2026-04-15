import numpy as np
import cv2
from pathlib import Path

EXTS = ("*.jpg", "*.png", "*.webp")

DEBUG = True
# DEBUG = False
if DEBUG:
    DEBUG_PATH = Path("debug/")
    DEBUG_PATH.mkdir(exist_ok=True)


def u8(img):
    if img is not None and img.dtype != np.uint8:
        img = cv2.convertScaleAbs(
            cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX)
        )
    return img


def pre_proc(img):
    img = u8(img)
    if img is not None:
        pass
    return img


def proc(path):
    org_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    pre_img = pre_proc(org_img)
    if pre_img is not None:
        pass
    return


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
            if DEBUG:
                F_NAME, F_EXT = name, ext
            res = proc(f)
            if res is not None:
                cv2.imwrite(str(out / f"{name}{ext}"), res)
            else:
                print(f"Failed to process {f}")
    else:
        print(f"No image files found in {inp.resolve()}")
