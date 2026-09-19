@echo off
title LTA Smart Depot AI Condition Monitoring System (PS3)
echo =====================================================================
echo  SINGAPORE LAND TRANSPORT AUTHORITY (LTA) SMART DEPOT OCC DASHBOARD
echo  NebulaX Hackathon Problem Statement 3 (PS3)
echo =====================================================================
echo.
echo Launching FastAPI Server on http://localhost:8080...
echo.

REM Try using py launcher, fallback to python
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    start http://localhost:8080
    py app.py
) else (
    start http://localhost:8080
    python app.py
)

pause
