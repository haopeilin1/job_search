@echo off
cd /d D:\git_vscode\job_search\backend
call venv\Scripts\activate.bat
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > logs\backend.log 2>&1
