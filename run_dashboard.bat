@echo off
echo Starting GridLock Sentry Backend API...
cd backend
start "GridLock Sentry Backend" cmd /k "python -m uvicorn gridlock_sentry_api:app --host 127.0.0.1 --port 8000"
cd ..
timeout /t 3 >nul
echo Opening GridLock Sentry Dashboard...
start frontend\index.html
echo Dashboard is now running!
echo You can close this window. Keep the backend window open.
pause