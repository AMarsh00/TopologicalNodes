@echo off
REM Build_Module_Windows.bat
REM Alexander Marsh
REM Last Edit 10 September 2025
REM ======================================================
REM
REM GNU Affero General Public License
REM
REM Build script for LooseTopologicalNode PyTorch extension. Run in x64 Visual Studio Developer Prompt.
REM ------------------------------------------------------
REM - Compiles the C++/CUDA extension via setup.py
REM - Places the .so file in the current directory
REM - Requires Python and PyTorch installed
REM ======================================================

echo Building LooseTopologicalNode extension...

REM Run the build script with the 'inplace' flag to output the .so file here
set DISTUTILS_USE_SDK=1
set MSSdk=1
python Setup.py build_ext --inplace

REM Check if the build succeeded
if %ERRORLEVEL% NEQ 0 (
    echo Build failed.
    exit /b %ERRORLEVEL%
) else (
    echo Build succeeded!
    echo The extension was built as LooseTopologicalNode*.pyd
    echo You can now import it in Python if it's in the same folder as your script.
)

pause
