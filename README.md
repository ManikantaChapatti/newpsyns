# PSYNS — College Letter Management System

## Run on Windows

### Backend terminal
From the project root:
```powershell
cd .\backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8001
```
Keep this terminal open. Backend docs: http://127.0.0.1:8001/docs

### Frontend terminal
Open a second terminal at the project root:
```powershell
cd .\frontend
npm install
npm run dev
```
Open the local URL Vite prints (usually http://localhost:5173).

## Demo dashboards
The home page includes Faculty Dashboard, Proctor Dashboard, and In-charge Dashboard buttons. Each opens a role-labelled demo workspace with overview statistics, all letter requests, status updates, and its own queue tab. These are demonstration entry points, **not secure role authentication**. Do not use demo access for real student data.

## Face matching limitations
The local face-matching feature compares a face found in the uploaded ID image with the submitted photo. The YuNet and SFace ONNX model files must be available in `backend/models/` with the expected filenames. If missing, obtain them from OpenCV Zoo. Face matching is not liveness detection and does not establish that an ID is genuine. Test matching and non-matching images before presenting.

## Important
This is a student project/demo and is not production-ready. Staff status-update endpoints do not have production-grade authentication/authorization. Do not claim liveness detection, ID authenticity, or secure faculty/proctor/in-charge authentication unless separately implemented and tested.


### Student letter file upload
Students can attach an optional supporting letter/document when submitting a request. Accepted formats: PDF, DOC, DOCX, JPG, PNG; maximum 10 MB. Uploaded attachments appear as download links in student and staff request tables. Files are stored locally under `backend/data/uploads/letters/`. This is a local demo; add authentication and access controls before production use.
