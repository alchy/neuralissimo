@echo off
REM Batch script for generating multiple piano samples
REM Usage: generate_batch.bat [model_path]

set MODEL=%1
if "%MODEL%"=="" set MODEL=src\checkpoints\best_model.pth

echo ========================================
echo Neurailssimo - Batch Sample Generator
echo ========================================
echo.
echo Model: %MODEL%
echo.

REM Create output directory
if not exist outputs mkdir outputs

REM Generate samples for selected MIDI notes and velocities
echo Generating samples...
echo.

REM Middle C (60) with various velocities
for %%v in (0 2 4 6) do (
    echo Generating: MIDI 60 Velocity %%v
    python src\inference.py --model %MODEL% --midi 60 --velocity %%v --output outputs\C4_vel%%v.wav
)

REM Various notes at medium velocity
for %%n in (48 52 55 60 64 67 72) do (
    echo Generating: MIDI %%n Velocity 4
    python src\inference.py --model %MODEL% --midi %%n --velocity 4 --output outputs\note_%%n_vel4.wav
)

REM Generate interpolations
echo.
echo Generating interpolations...
python src\inference.py --model %MODEL% --interpolate --midi 48 --velocity 1 --midi2 72 --velocity2 7 --num-steps 8 --output outputs\interp.wav

echo.
echo ========================================
echo Batch generation complete!
echo Outputs saved to: outputs\
echo ========================================
pause
