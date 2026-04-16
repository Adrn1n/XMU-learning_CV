from pathlib import Path
import cv2
import numpy as np

OUTPUT_PATH = Path("output/")
MODEL_PATH = Path("models/")
RECT_COLOR = (0, 255, 0)
RECT_THICKNESS = 2
FONT_Y_OFFSET = 5
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)
FONT_THICKNESS = 1
COSINE_THRESHOLD = 0.363
RECT_COLOR_UNKNOWN = (0, 0, 255)
FONT_COLOR_UNKNOWN = (0, 0, 255)
EXTS = ("*.jpg", "*.png", "*.webp")
GALLERY_PATH = Path("gallery/")
GALLERY_OUTPUT_PATH = OUTPUT_PATH / "gallery/"
INPUT_PATH = Path("input/")

face_detector = cv2.FaceDetectorYN_create(
    str(MODEL_PATH / "face_detection_yunet_2023mar.onnx"), "", (0, 0)
)
face_recognizer = cv2.FaceRecognizerSF_create(
    str(MODEL_PATH / "face_recognition_sface_2021dec.onnx"), ""
)


def pre_proc(path, gallery_dict, detector, recognizer):
    if gallery_dict is None:
        gallery_dict = {}
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = detector.detect(img)
        if faces is not None and len(faces) > 0:
            for i, face in enumerate(faces):
                gallery_dict[path.stem + path.suffix + f"_{i}"] = recognizer.feature(
                    recognizer.alignCrop(img, face)
                ).flatten()
                x, y, w, h = map(int, face[:4])
                cv2.rectangle(img, (x, y), (x + w, y + h), RECT_COLOR, RECT_THICKNESS)
                cv2.putText(
                    img,
                    f"{i}",
                    (x, y - FONT_Y_OFFSET),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE,
                    FONT_COLOR,
                    FONT_THICKNESS,
                )
        else:
            img = None
    return gallery_dict, img


def proc(path, gallery_dict, detector, recognizer):
    img = None
    if gallery_dict is not None and len(gallery_dict) > 0:
        org_img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if org_img is not None:
            detector.setInputSize((org_img.shape[1], org_img.shape[0]))
            _, faces = detector.detect(org_img)
            if faces is not None and len(faces) > 0:
                img = org_img
                for face in faces:
                    feature = recognizer.feature(
                        recognizer.alignCrop(org_img, face)
                    ).flatten()
                    tag, sim = None, -1
                    for k, v in gallery_dict.items():
                        tmp = np.dot(feature, v) / (
                            (np.linalg.norm(feature) * np.linalg.norm(v))
                        )
                        if tmp > sim:
                            tag = k
                            sim = tmp
                    x, y, w, h = map(int, face[:4])
                    cv2.rectangle(
                        img,
                        (x, y),
                        (x + w, y + h),
                        RECT_COLOR if sim >= COSINE_THRESHOLD else RECT_COLOR_UNKNOWN,
                        RECT_THICKNESS,
                    )
                    cv2.putText(
                        img,
                        (
                            f"{tag} ({sim:.2f})"
                            if sim >= COSINE_THRESHOLD
                            else f"Unknown ({sim:.2f})"
                        ),
                        (x, y - FONT_Y_OFFSET),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE,
                        FONT_COLOR if sim >= COSINE_THRESHOLD else FONT_COLOR_UNKNOWN,
                        FONT_THICKNESS,
                    )
    return img


if __name__ == "__main__":
    gallery = {}
    for ext in EXTS:
        for g_path in GALLERY_PATH.glob(ext):
            gallery, res = pre_proc(g_path, gallery, face_detector, face_recognizer)
            if res is not None:
                GALLERY_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(
                    str(GALLERY_OUTPUT_PATH / f"{g_path.stem}{g_path.suffix}"), res
                )
            else:
                print(f"Failed to build gallery for {g_path}")
    if gallery is not None and len(gallery) > 0:
        for ext in EXTS:
            for g_path in INPUT_PATH.glob(ext):
                res = proc(g_path, gallery, face_detector, face_recognizer)
                if res is not None:
                    cv2.imwrite(str(OUTPUT_PATH / f"{g_path.stem}{g_path.suffix}"), res)
                else:
                    print(f"Failed to process {g_path}")
    else:
        print("No valid gallery features extracted")
