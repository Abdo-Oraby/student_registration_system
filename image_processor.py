"""
image_processor.py
Handles:
  - Face detection & smart auto-crop centred on face
  - Resize to standard dimensions (400×500 portrait)
  - Quality compression
  - Manual rotation/flip support
  - Cloudinary upload (optional)
"""
import os, io, re
import cv2
import numpy as np
from PIL import Image, ImageOps, ExifTags

try:
    import cloudinary
    import cloudinary.uploader
    _CLD = bool(os.getenv("CLOUDINARY_CLOUD_NAME"))
    if _CLD:
        cloudinary.config(
            cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key    = os.getenv("CLOUDINARY_API_KEY"),
            api_secret = os.getenv("CLOUDINARY_API_SECRET"),
        )
except ImportError:
    _CLD = False

TARGET_W   = 400
TARGET_H   = 500
JPEG_Q     = 88          # output quality
CASCADES   = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",
    cv2.data.haarcascades + "haarcascade_profileface.xml",
]


# ── helpers ────────────────────────────────────────────────────────────────

def _fix_exif_rotation(pil_img: Image.Image) -> Image.Image:
    """Auto-rotate image based on EXIF orientation tag."""
    try:
        return ImageOps.exif_transpose(pil_img)
    except Exception:
        return pil_img


def _bytes_to_cv(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _cv_to_bytes(img: np.ndarray, quality: int = JPEG_Q) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else b""


def _pil_to_bytes(img: Image.Image, quality: int = JPEG_Q) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def detect_faces(image_bytes: bytes) -> list[tuple]:
    """
    Return list of (x,y,w,h) face rectangles.
    Uses 4 cascades + histogram equalisation for robustness.
    """
    img = _bytes_to_cv(image_bytes)
    if img is None:
        return []

    h, w = img.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img   = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    for path in CASCADES:
        if not os.path.exists(path):
            continue
        cascade = cv2.CascadeClassifier(path)
        faces   = cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30)
        )
        if len(faces) > 0:
            # scale back if we down-sampled
            orig_h, orig_w = _bytes_to_cv(image_bytes).shape[:2]
            sx = orig_w / img.shape[1]
            sy = orig_h / img.shape[0]
            return [(int(x*sx), int(y*sy), int(w*sx), int(h*sy)) for (x,y,w,h) in faces]

    return []


def smart_crop_face(image_bytes: bytes) -> bytes:
    """
    Auto-crop the image centred on the largest detected face.
    Adds headroom above and body below.
    Returns 400×500 JPEG bytes.
    If no face found, returns a centred crop.
    """
    pil = Image.open(io.BytesIO(image_bytes))
    pil = _fix_exif_rotation(pil)
    pil = pil.convert("RGB")

    faces = detect_faces(image_bytes)
    iw, ih = pil.size

    if faces:
        # Pick largest face
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2

        # Desired crop height = face_h × 3.5  (head + body)
        crop_h = int(fh * 3.5)
        crop_w = int(crop_h * TARGET_W / TARGET_H)  # maintain aspect ratio

        # Position: face centre at ~28% from top of crop
        top  = face_cy - int(crop_h * 0.28)
        left = face_cx - crop_w // 2

        # Clamp to image bounds
        top  = max(0, min(top,  ih - crop_h))
        left = max(0, min(left, iw - crop_w))
        # If crop larger than image, fall back to full image
        if crop_w > iw or crop_h > ih:
            top, left, crop_w, crop_h = 0, 0, iw, ih
    else:
        # No face – centre crop at target ratio
        ar      = TARGET_W / TARGET_H
        if iw / ih > ar:
            crop_w = int(ih * ar)
            crop_h = ih
        else:
            crop_h = int(iw / ar)
            crop_w = iw
        left = (iw - crop_w) // 2
        top  = (ih - crop_h) // 2

    cropped = pil.crop((left, top, left + crop_w, top + crop_h))
    resized = cropped.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    return _pil_to_bytes(resized)


def apply_edits(image_bytes: bytes,
                rotation: int = 0,
                flip_h:   bool = False,
                zoom:     float = 1.0,
                offset_x: float = 0.0,
                offset_y: float = 0.0,
                auto_crop: bool = True) -> bytes:
    """
    Apply manual edits from the front-end canvas editor, then
    optionally run smart face-crop + resize.

    rotation  : degrees CW (0/90/180/270)
    flip_h    : horizontal mirror
    zoom      : 1.0 = fit, >1 = zoom in
    offset_x/y: pan fraction (-1..1) relative to half-size
    auto_crop : run smart_crop_face afterwards
    """
    pil = Image.open(io.BytesIO(image_bytes))
    pil = _fix_exif_rotation(pil).convert("RGB")

    # Apply flip
    if flip_h:
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)

    # Apply rotation (PIL rotates CCW; negate for CW)
    if rotation:
        pil = pil.rotate(-rotation, expand=True)

    # Apply zoom + pan
    if zoom != 1.0 or offset_x != 0.0 or offset_y != 0.0:
        iw, ih   = pil.size
        new_w    = int(iw / zoom)
        new_h    = int(ih / zoom)
        cx       = iw // 2 + int(offset_x * iw * 0.5)
        cy       = ih // 2 + int(offset_y * ih * 0.5)
        left     = max(0, cx - new_w // 2)
        top      = max(0, cy - new_h // 2)
        left     = min(left, iw - new_w)
        top      = min(top,  ih - new_h)
        pil      = pil.crop((left, top, left + new_w, top + new_h))

    edited_bytes = _pil_to_bytes(pil)

    if auto_crop:
        return smart_crop_face(edited_bytes)

    # No auto-crop: just resize to target
    pil = pil.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    return _pil_to_bytes(pil)


def _college_folder(college: str) -> str:
    """Return a safe folder name for the college (Arabic-friendly, spaces→underscore)."""
    # Keep Arabic letters/digits, replace spaces/special chars with _
    safe = re.sub(r'[\s/\\:*?"<>|]+', '_', college.strip())
    return safe.rstrip('_') or "عام"


# ── storage ────────────────────────────────────────────────────────────────

def save_image(image_bytes: bytes, student_id: str, year: str,
               college: str, upload_root: str) -> dict:
    """
    Save processed image.
    Local path: uploads/{year}/{college_folder}/{student_id}.jpg
    Returns { "path": relative_path, "url": public_url }
    """
    if _CLD:
        col_slug = _college_folder(college)
        result = cloudinary.uploader.upload(
            image_bytes,
            public_id     = f"students/{year}/{col_slug}/{student_id}",
            overwrite     = True,
            resource_type = "image",
            transformation = [{"width": TARGET_W, "height": TARGET_H,
                               "crop": "fill", "gravity": "face"}]
        )
        return {"path": result["public_id"], "url": result["secure_url"], "cloudinary": True}
    else:
        col_folder = _college_folder(college)
        # uploads/{year}/{college}/{student_id}.jpg
        folder     = os.path.join(upload_root, year, col_folder)
        os.makedirs(folder, exist_ok=True)
        filename   = f"{student_id}.jpg"
        full_path  = os.path.join(folder, filename)
        with open(full_path, "wb") as f:
            f.write(image_bytes)
        # relative path from static/
        rel_path = os.path.relpath(full_path,
                                   os.path.join(upload_root, "..")).replace("\\", "/")
        return {"path": rel_path, "url": f"/static/{rel_path}", "cloudinary": False}


def archive_old_image(old_rel_path: str, student_id: str,
                      static_root: str, upload_root: str) -> str | None:
    """
    Rename old image to {id}_old_{timestamp}.jpg in same folder.
    Returns the new relative path (or None).
    """
    from datetime import datetime
    old_full = os.path.join(static_root, old_rel_path)
    if not os.path.exists(old_full):
        return None
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    old_dir  = os.path.dirname(old_full)
    arc_name = f"{student_id}_old_{ts}.jpg"
    arc_full = os.path.join(old_dir, arc_name)
    os.rename(old_full, arc_full)
    return os.path.relpath(arc_full, static_root).replace("\\", "/")
