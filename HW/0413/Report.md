# Report 3
## Requirements
- Implement a complete face recognition program that can identify several dormitory classmates.
    - Face detection
    - Face feature extraction & comparison
- Submission format: Python code + (2~3 page) report
- Submit within 2 weeks

- Reference
    - [https://docs.opencv.org/4.9.0/d0/dd4/tutorial_dnn_face.html](https://docs.opencv.org/4.9.0/d0/dd4/tutorial_dnn_face.html)
    - Face detector: `cv2.FaceDetectorYN`
    - Face feature extract: `cv2.FaceRecognizerSF`

## Implementation
### Methodology

### Overview
```mermaid
graph TD
    Ga["Gallery"] --> |"YUNet Detection <br> AlignCrop + SFace Feature"| Dict["Gallery Feature Dictionary(Name: Feature)"]
    In["Input"] --> |"YUNet Detection <br> AlignCrop + SFace Feature"| QF["Query Features"]
    Dict & QF -.-> M(("Cosine Similarity <br> Best Match + Score"))
    M --> |"score >= Threshold"| K["Label = Name <br> Green Box + Score"]
    M --> |"score < Threshold"| U["Label = Unknown <br> Red Box + Score"]
    K & U --> |"Draw Rectangle + Text"| Out["Output"]
```

### Parameters

### Features

## Code
```python
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


def pre_proc(path, gallery, detector, recognizer):
    if gallery is None:
        gallery = {}
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is not None:
        detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = detector.detect(img)
        if faces is not None and len(faces) > 0:
            for i, face in enumerate(faces):
                gallery[path.stem + path.suffix + f"_{i}"] = recognizer.feature(
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
    return gallery, img


def proc(path, gallery, detector, recognizer):
    img = None
    if gallery is not None and len(gallery) > 0:
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
                    for k, v in gallery.items():
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
        for path in GALLERY_PATH.glob(ext):
            gallery, img = pre_proc(path, gallery, face_detector, face_recognizer)
            if img is not None:
                GALLERY_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
                cv2.imwrite(str(GALLERY_OUTPUT_PATH / f"{path.stem}{path.suffix}"), img)
            else:
                print(f"Failed to build gallery for {path}")
    if gallery is not None and len(gallery) > 0:
        for ext in EXTS:
            for path in INPUT_PATH.glob(ext):
                res = proc(path, gallery, face_detector, face_recognizer)
                if res is not None:
                    cv2.imwrite(str(OUTPUT_PATH / f"{path.stem}{path.suffix}"), res)
                else:
                    print(f"Failed to process {path}")
    else:
        print("No valid gallery features extracted")

```

## Results
