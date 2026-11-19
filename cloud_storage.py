"""Cloud and In-Record Image Storage Handler for CivicFusion AI.

Provides resilient, multi-tiered image persistence:
1. ImgBB Cloud CDN (permanent HTTPS URLs via free API key)
2. In-Record Compressed Base64 (zero-dependency memory persistence)
3. Local Disk Storage with Linux/Windows POSIX path normalization
"""

import base64
import io
import os
from pathlib import Path
from typing import Any, Optional
from PIL import Image
import requests

IMGBB_UPLOAD_URL = "https://api.imgbb.com/1/upload"


def get_active_imgbb_key() -> Optional[str]:
    """Retrieve ImgBB API key from environment, streamlit secrets, or return None."""
    # 1. Environment variable (e.g. on Render)
    key = os.getenv("IMGBB_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. Streamlit secrets if running in Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "IMGBB_API_KEY" in st.secrets:
            sec_key = st.secrets["IMGBB_API_KEY"]
            if sec_key and str(sec_key).strip():
                return str(sec_key).strip()
        # 3. Officer session override if entered in UI
        if hasattr(st, "session_state") and "imgbb_api_key_override" in st.session_state:
            override = st.session_state["imgbb_api_key_override"]
            if override and str(override).strip():
                return str(override).strip()
    except Exception:
        pass

    return None


def compress_image_to_jpeg_bytes(
    image: Image.Image,
    max_dimension: int = 1200,
    quality: int = 80
) -> bytes:
    """Resize and compress a PIL Image to lightweight JPEG bytes (~60-120 KB)."""
    img = image.copy()
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dimension:
        if w > h:
            new_w = max_dimension
            new_h = int(h * (max_dimension / w))
        else:
            new_h = max_dimension
            new_w = int(w * (max_dimension / h))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def upload_to_imgbb(
    image_bytes: bytes,
    filename: str = "evidence.jpg",
    api_key: Optional[str] = None
) -> Optional[str]:
    """Upload JPEG bytes to ImgBB and return permanent HTTPS image URL, or None on failure."""
    active_key = api_key or get_active_imgbb_key()
    if not active_key:
        return None

    try:
        b64_data = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "key": active_key,
            "image": b64_data,
            "name": filename.replace(".jpg", "")
        }
        resp = requests.post(IMGBB_UPLOAD_URL, data=payload, timeout=12)
        if resp.status_code == 200:
            res_json = resp.json()
            if res_json.get("success") and "data" in res_json:
                # Prefer direct image URL (e.g. https://i.ibb.co/xyz/image.jpg)
                return res_json["data"].get("url") or res_json["data"].get("display_url")
    except Exception as exc:
        print(f"[CloudStorage] ImgBB upload notice: {exc}")

    return None


def store_complaint_image(
    image: Image.Image,
    ticket_id: str,
    is_annotated: bool = False,
    api_key: Optional[str] = None
) -> dict[str, Any]:
    """Compress image, save locally (POSIX path), generate Base64, and upload to ImgBB."""
    suffix = "_annotated" if is_annotated else ""
    filename = f"{ticket_id}{suffix}.jpg"

    # 1. Compress to JPEG
    jpeg_bytes = compress_image_to_jpeg_bytes(image)

    # 2. Generate Base64 Data URI
    b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_str}"

    # 3. Save locally with POSIX-normalized forward slashes
    upload_dir = Path(__file__).parent / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    local_path = upload_dir / filename
    try:
        local_path.write_bytes(jpeg_bytes)
        posix_rel_path = f"data/uploads/{filename}"
    except Exception:
        posix_rel_path = None

    # 4. Attempt Cloud CDN upload
    cloud_url = upload_to_imgbb(jpeg_bytes, filename=filename, api_key=api_key)

    return {
        "url": cloud_url,
        "path": posix_rel_path,
        "base64": data_uri,
    }


def load_pil_image_from_complaint(
    complaint: dict[str, Any],
    is_annotated: bool = False
) -> Optional[Image.Image]:
    """Retrieve a PIL Image from URL, Base64, or local path with graceful fallback."""
    url_key = "annotated_photo_url" if is_annotated else "photo_url"
    b64_key = "annotated_photo_base64" if is_annotated else "photo_base64"
    path_key = "annotated_photo_path" if is_annotated else "photo_path"

    # 1. Check Cloud URL
    url = complaint.get(url_key)
    if url and isinstance(url, str) and url.startswith("http"):
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            pass

    # 2. Check Base64 URI
    b64_val = complaint.get(b64_key)
    if b64_val and isinstance(b64_val, str):
        try:
            if "," in b64_val:
                b64_clean = b64_val.split(",", 1)[1]
            else:
                b64_clean = b64_val
            raw_bytes = base64.b64decode(b64_clean)
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception:
            pass

    # 3. Check Local File Path (normalize forward slashes for Linux)
    p_val = complaint.get(path_key)
    if p_val and isinstance(p_val, str):
        clean_p = p_val.replace("\\", "/")
        # Try relative to cwd or module
        candidates = [
            Path(clean_p),
            Path(__file__).parent / clean_p,
            Path(__file__).parent / "data" / "uploads" / Path(clean_p).name,
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                try:
                    return Image.open(c).convert("RGB")
                except Exception:
                    pass

    # 4. Fallback check for standard {id}.jpg in data/uploads
    cid = complaint.get("id")
    if cid:
        suf = "_annotated.jpg" if is_annotated else ".jpg"
        fallback_file = Path(__file__).parent / "data" / "uploads" / f"{cid}{suf}"
        if fallback_file.exists():
            try:
                return Image.open(fallback_file).convert("RGB")
            except Exception:
                pass

    return None
