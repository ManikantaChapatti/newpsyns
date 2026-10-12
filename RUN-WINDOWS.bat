@echo off
echo Open two terminals as described in README.md.
echo Backend: cd backend ^&^& py -m pip install -r requirements.txt ^&^& py -m uvicorn main:app --reload --port 8001
echo Frontend: cd frontend ^&^& npm install ^&^& npm run dev
pause
