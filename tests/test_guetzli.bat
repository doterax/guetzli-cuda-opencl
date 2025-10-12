@echo off
setlocal enabledelayedexpansion

echo Starting Guetzli CUDA and OpenCL tests...
echo.

REM Create output directory if it doesn't exist
if not exist "output" mkdir output

REM Set the path to guetzli.exe (relative to tests folder)
set GUETZLI_PATH=..\bin\x86_64\Release\guetzli.exe

REM Check if guetzli.exe exists
if not exist "%GUETZLI_PATH%" (
    echo ERROR: guetzli.exe not found at %GUETZLI_PATH%
    exit /b 1
)

echo Processing input files...
echo.

REM Process each file in the input directory
for %%f in (input\*.*) do (
    set "input_file=%%f"
    set "filename=%%~nf"
    set "extension=%%~xf"
    
    echo Processing: !input_file!
    
    REM Generate CUDA output
    set "cuda_output=output\!filename!_cuda!extension!"
    echo   CUDA: !cuda_output!
    "%GUETZLI_PATH%" --cuda "!input_file!" "!cuda_output!"
    if errorlevel 1 (
        echo   ERROR: CUDA processing failed for !input_file!
    )
    
    REM Generate OpenCL output
    set "opencl_output=output\!filename!_opencl!extension!"
    echo   OpenCL: !opencl_output!
    "%GUETZLI_PATH%" --opencl "!input_file!" "!opencl_output!"
    if errorlevel 1 (
        echo   ERROR: OpenCL processing failed for !input_file!
    )
    
    echo.
)

echo All files processed!
echo.
echo Test completed!
pause
