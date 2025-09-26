@echo off
setlocal enabledelayedexpansion

echo Starting Guetzli CUDA and OpenCL tests with hash comparison...
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

REM Generate MD5 hashes and compare CUDA vs OpenCL
echo Generating MD5 hashes and comparing CUDA vs OpenCL...
echo.

REM Create hash comparison file
set "hash_file=output\hash_comparison.txt"
echo MD5 Hash Comparison: CUDA vs OpenCL > "%hash_file%"
echo ====================================== >> "%hash_file%"
echo. >> "%hash_file%"

set "identical_count=0"
set "different_count=0"

for %%f in (input\*.*) do (
    set "filename=%%~nf"
    set "extension=%%~xf"
    
    set "cuda_file=output\!filename!_cuda!extension!"
    set "opencl_file=output\!filename!_opencl!extension!"
    
    if exist "!cuda_file!" if exist "!opencl_file!" (
        echo File: !filename!!extension! >> "%hash_file%"
        echo ---------------------------------------- >> "%hash_file%"
        
        REM Get CUDA hash
        for /f "tokens=1" %%a in ('certutil -hashfile "!cuda_file!" MD5 ^| findstr /v ":"') do set "cuda_hash=%%a"
        
        REM Get OpenCL hash
        for /f "tokens=1" %%a in ('certutil -hashfile "!opencl_file!" MD5 ^| findstr /v ":"') do set "opencl_hash=%%a"
        
        echo CUDA:   !cuda_hash! >> "%hash_file%"
        echo OpenCL: !opencl_hash! >> "%hash_file%"
        
        if "!cuda_hash!"=="!opencl_hash!" (
            echo RESULT: IDENTICAL >> "%hash_file%"
            set /a identical_count+=1
        ) else (
            echo RESULT: DIFFERENT >> "%hash_file%"
            set /a different_count+=1
        )
        echo. >> "%hash_file%"
    )
)

echo Summary: >> "%hash_file%"
echo ======== >> "%hash_file%"
echo Identical outputs: !identical_count! >> "%hash_file%"
echo Different outputs: !different_count! >> "%hash_file%"

echo Hash comparison saved to: %hash_file%
echo.

REM Display the comparison results
type "%hash_file%"

echo.
echo Test completed!
echo Identical outputs: !identical_count!
echo Different outputs: !different_count!
pause
