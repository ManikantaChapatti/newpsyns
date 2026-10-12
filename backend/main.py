from pathlib import Path
from uuid import uuid4
import json
import hashlib
import secrets
import urllib.request

import cv2
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
LETTER_UPLOADS = UPLOADS / "letters"
DATA.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)
LETTER_UPLOADS.mkdir(parents=True, exist_ok=True)
USERS_FILE = DATA / "users.json"
LETTERS_FILE = DATA / "letters.json"
MAX_FILE_SIZE = 5 * 1024 * 1024
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
MODEL_URLS = {
    YUNET_PATH: "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    SFACE_PATH: "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def password_hash(password: str, salt: str | None = None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return salt, digest

app = FastAPI(title="PSYNS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RegisterBody(BaseModel):
    roll_number: str = Field(min_length=3, max_length=40)
    full_name: str = Field(min_length=2, max_length=100)
    department: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=8, max_length=128)

class LoginBody(BaseModel):
    roll_number: str
    password: str

class LetterBody(BaseModel):
    student_name: str
    roll_number: str
    department: str
    letter_type: str
    purpose: str = Field(min_length=5, max_length=1000)
    recipient: str = "Academic Office"

@app.get("/")
def home():
    return {"message": "PSYNS backend is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/register-student")
def register_student(body: RegisterBody):
    users = read_json(USERS_FILE, [])
    roll = body.roll_number.strip().upper()
    if any(u["roll_number"] == roll for u in users):
        raise HTTPException(status_code=409, detail="This roll number is already registered.")
    salt, digest = password_hash(body.password)
    users.append({
        "roll_number": roll, "full_name": body.full_name.strip(),
        "department": body.department.strip(), "salt": salt, "password_hash": digest,
        "role": "student"
    })
    write_json(USERS_FILE, users)
    return {"success": True, "message": "Account created. Please sign in."}

@app.post("/login-student")
def login_student(body: LoginBody):
    users = read_json(USERS_FILE, [])
    roll = body.roll_number.strip().upper()
    user = next((u for u in users if u["roll_number"] == roll), None)
    if not user:
        raise HTTPException(status_code=401, detail="Roll number or password is incorrect.")
    _, digest = password_hash(body.password, user["salt"])
    if not secrets.compare_digest(digest, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Roll number or password is incorrect.")
    return {"success": True, "user": {"roll_number": user["roll_number"], "full_name": user["full_name"], "department": user["department"], "role": user["role"]}}

async def save_image(upload: UploadFile, prefix: str) -> str:
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG, or WEBP image.")
    content = await upload.read(MAX_FILE_SIZE + 1)
    if not content or len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Image must be between 1 byte and 5 MB.")
    filename = f"{prefix}_{uuid4().hex}.img"
    destination = UPLOADS / filename
    try:
        destination.write_bytes(content)
    except OSError:
        raise HTTPException(status_code=500, detail="Could not save uploaded image.")
    return str(destination.relative_to(BASE))

@app.post("/verification/upload-id")
async def upload_id_card(user_id: str = Form(...), id_card: UploadFile = File(...)):
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="College ID or staff ID is required.")
    saved = await save_image(id_card, "id")
    return {"message": "ID image uploaded.", "user_id": user_id.strip(), "upload_received": True, "file": saved}

def ensure_face_models():
    """Download the open-source OpenCV Zoo models on first use."""
    for path, url in MODEL_URLS.items():
        if path.exists() and path.stat().st_size > 100_000:
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PSYNS-local-demo"})
            with urllib.request.urlopen(request, timeout=45) as response, path.open("wb") as output:
                output.write(response.read())
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail=f"Could not download face model {path.name}. Check internet and restart verification. Details: {exc}")


def image_from_upload(data: bytes):
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="One of the uploaded files is not a readable image.")
    return image


def get_face_embedding(image, detector, recognizer):
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    result = detector.detect(image)
    faces = result[1]
    if faces is None or len(faces) == 0:
        return None
    # Use the largest detected face, which is typically the ID portrait or webcam face.
    face = max(faces, key=lambda row: float(row[2] * row[3]))
    aligned = recognizer.alignCrop(image, face)
    return recognizer.feature(aligned)


@app.post("/verification/verify")
async def verify_identity(
    user_id: str = Form(...),
    id_card: UploadFile = File(...),
    live_photo: UploadFile = File(...)
):
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="College ID or staff ID is required.")
    if id_card.content_type not in ALLOWED_TYPES or live_photo.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG, or WEBP image.")
    id_bytes = await id_card.read(MAX_FILE_SIZE + 1)
    live_bytes = await live_photo.read(MAX_FILE_SIZE + 1)
    if not id_bytes or not live_bytes or len(id_bytes) > MAX_FILE_SIZE or len(live_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Each image must be between 1 byte and 5 MB.")
    ensure_face_models()
    try:
        detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.7, 0.3, 5000)
        recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
        id_image = image_from_upload(id_bytes)
        live_image = image_from_upload(live_bytes)
        id_embedding = get_face_embedding(id_image, detector, recognizer)
        live_embedding = get_face_embedding(live_image, detector, recognizer)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Face analysis failed. Use clear, front-facing images. Details: {exc}")
    if id_embedding is None:
        raise HTTPException(status_code=422, detail="No face was detected on the college ID. Upload a clear image where the ID portrait is visible.")
    if live_embedding is None:
        raise HTTPException(status_code=422, detail="No face was detected in the live photo. Capture a clear, front-facing photo with good lighting.")
    score = float(recognizer.match(id_embedding, live_embedding, cv2.FaceRecognizerSF_FR_COSINE))
    threshold = 0.363
    verified = score >= threshold
    # Save only the images used for this comparison, after successful processing.
    id_path = UPLOADS / f"verified_id_{uuid4().hex}.img"
    live_path = UPLOADS / f"verified_live_{uuid4().hex}.img"
    id_path.write_bytes(id_bytes)
    live_path.write_bytes(live_bytes)
    return {
        "verified": verified,
        "demo_mode": False,
        "method": "OpenCV YuNet face detection + SFace embedding similarity",
        "similarity_score": round(score, 4),
        "threshold": threshold,
        "liveness_checked": False,
        "message": ("Face match passed. This comparison does not verify ID authenticity or liveness." if verified else "Face match failed. Use a clear ID portrait and a well-lit, front-facing live photo. This comparison does not verify liveness."),
        "user_id": user_id.strip(),
        "id_image": str(id_path.relative_to(BASE)),
        "live_image": str(live_path.relative_to(BASE))
    }

@app.get("/teachers")
def teachers():
    return [
        {"id": "faculty-1", "name": "Academic Office", "department": "Administration"},
        {"id": "faculty-2", "name": "Department Coordinator", "department": "Faculty"},
        {"id": "faculty-3", "name": "Technical In-charge", "department": "IT"}
    ]

@app.get("/letters")
def list_letters():
    return read_json(LETTERS_FILE, [])

@app.post("/letters")
def create_letter(body: LetterBody):
    letters = read_json(LETTERS_FILE, [])
    item = body.model_dump()
    item.update({"id": uuid4().hex[:10].upper(), "status": "Pending", "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds")})
    letters.insert(0, item)
    write_json(LETTERS_FILE, letters)
    return {"success": True, "letter": item}

LETTER_ALLOWED_TYPES = {
    "application/pdf", "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/jpeg", "image/png"
}
LETTER_MAX_SIZE = 10 * 1024 * 1024

@app.post("/letters/submit")
async def submit_letter_with_attachment(
    student_name: str = Form(...),
    roll_number: str = Form(...),
    department: str = Form(...),
    letter_type: str = Form(...),
    purpose: str = Form(...),
    recipient: str = Form("Academic Office"),
    attachment: UploadFile | None = File(None),
):
    if len(purpose.strip()) < 5:
        raise HTTPException(status_code=400, detail="Please enter at least 5 characters for the purpose.")
    attachment_info = None
    if attachment and attachment.filename:
        if attachment.content_type not in LETTER_ALLOWED_TYPES:
            raise HTTPException(status_code=400, detail="Attach a PDF, DOC, DOCX, JPG, or PNG file.")
        content = await attachment.read(LETTER_MAX_SIZE + 1)
        if not content or len(content) > LETTER_MAX_SIZE:
            raise HTTPException(status_code=400, detail="The attachment must be between 1 byte and 10 MB.")
        suffix = Path(attachment.filename).suffix.lower()
        safe_name = Path(attachment.filename).name.replace("\\", "_").replace("/", "_")[:120]
        stored_name = f"letter_{uuid4().hex}{suffix}"
        (LETTER_UPLOADS / stored_name).write_bytes(content)
        attachment_info = {"filename": safe_name, "stored_name": stored_name, "size_bytes": len(content)}
    letters = read_json(LETTERS_FILE, [])
    item = {
        "student_name": student_name.strip(), "roll_number": roll_number.strip().upper(),
        "department": department.strip(), "letter_type": letter_type.strip(),
        "purpose": purpose.strip(), "recipient": recipient.strip() or "Academic Office",
        "id": uuid4().hex[:10].upper(), "status": "Pending",
        "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "attachment": attachment_info,
    }
    letters.insert(0, item)
    write_json(LETTERS_FILE, letters)
    return {"success": True, "letter": item}

@app.get("/letters/{letter_id}/attachment")
def download_letter_attachment(letter_id: str):
    letters = read_json(LETTERS_FILE, [])
    item = next((x for x in letters if x["id"] == letter_id), None)
    attachment = item.get("attachment") if item else None
    if not attachment:
        raise HTTPException(status_code=404, detail="No attachment was uploaded for this letter.")
    path = LETTER_UPLOADS / Path(attachment["stored_name"]).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The attachment file could not be found.")
    return FileResponse(path, filename=attachment.get("filename", path.name))

@app.patch("/letters/{letter_id}/status")
def update_letter(letter_id: str, status: str = Form(...)):
    allowed = {"Pending", "Under Review", "Approved", "Rejected", "Completed"}
    if status not in allowed:
        raise HTTPException(status_code=400, detail="Invalid status.")
    letters = read_json(LETTERS_FILE, [])
    item = next((x for x in letters if x["id"] == letter_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Letter not found.")
    item["status"] = status
    write_json(LETTERS_FILE, letters)
    return {"success": True, "letter": item}
