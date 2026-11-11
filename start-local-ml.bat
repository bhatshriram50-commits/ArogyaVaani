@echo off
title ArogyaVaani - Local Hospital ML Service
echo =======================================================
echo Starting ArogyaVaani Local Hospital ML Service
echo Running on: http://127.0.0.1:8000
echo Raw medical images stay strictly on this computer!
echo =======================================================
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
python ml\service.py
pause

