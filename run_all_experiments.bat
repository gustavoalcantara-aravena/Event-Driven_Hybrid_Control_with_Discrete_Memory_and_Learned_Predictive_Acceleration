@echo off
echo ============================================================
echo   EVENT-DRIVEN HYBRID CONTROL: FULL EXPERIMENTAL SUITE
echo ============================================================
echo.

echo [1/5] Running Baseline Experiments (15 seeds, 25 scenarios)...
python run_baselines.py --plant both --seeds 15 --scenarios 25
if %ERRORLEVEL% NEQ 0 (
    echo Error in run_baselines.py
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [2/5] Running Proposed Method Experiments (15 seeds, 25 scenarios)...
python run_proposed.py --plant both --seeds 15 --scenarios 25
if %ERRORLEVEL% NEQ 0 (
    echo Error in run_proposed.py
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/5] Capturing Baseline Trajectories for Fig6...
python capture_trajectories_baselines.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in capture_trajectories_baselines.py
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [4/5] Capturing Proposed Trajectories for Fig6...
python capture_trajectories_proposed.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in capture_trajectories_proposed.py
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [5/5] Generating Final Tables and Figures...
python evaluate.py --plants "motor, oven"
if %ERRORLEVEL% NEQ 0 (
    echo Error in evaluate.py
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   EXPERIMENTS COMPLETE! Check the 'evaluation/' folder.
echo ============================================================
pause
