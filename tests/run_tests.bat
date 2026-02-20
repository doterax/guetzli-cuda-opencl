@echo off
setlocal enabledelayedexpansion
REM Guetzli CUDA/OpenCL - Test Suite (Windows)
REM Usage: tests\run_tests.bat <path-to-guetzli.exe>

if "%~1"=="" (
    echo Usage: %0 ^<guetzli.exe^>
    exit /b 1
)

set "GUETZLI=%~1"
set "SCRIPT_DIR=%~dp0"
set "TMPDIR=%TEMP%\guetzli_test_%RANDOM%"
mkdir "%TMPDIR%" 2>NUL

set PASS=0
set FAIL=0
set TOTAL=0

echo ============================================
echo   Guetzli CUDA/OpenCL - Test Suite
echo ============================================
echo Binary: %GUETZLI%
echo.

REM Test 1: Binary exists
set /a TOTAL+=1
echo|set /p="Test 1: Binary exists ............ "
if exist "%GUETZLI%" (
    echo PASS
    set /a PASS+=1
) else (
    echo FAIL: binary not found
    set /a FAIL+=1
)

REM Test 2: --version flag
set /a TOTAL+=1
echo|set /p="Test 2: --version ................ "
"%GUETZLI%" --version >NUL 2>&1
if %ERRORLEVEL% LSS 128 (
    echo PASS
    set /a PASS+=1
) else (
    echo FAIL: exit code %ERRORLEVEL%
    set /a FAIL+=1
)

REM Test 3: PNG to JPEG conversion
set /a TOTAL+=1
echo|set /p="Test 3: PNG to JPEG .............. "
set "INPUT_PNG=%SCRIPT_DIR%input\fro_small.png"
set "OUTPUT_JPG=%TMPDIR%\fro_small_out.jpg"
if not exist "%INPUT_PNG%" (
    echo FAIL: test input not found
    set /a FAIL+=1
    goto :test4skip
)
"%GUETZLI%" --quality 84 "%INPUT_PNG%" "%OUTPUT_JPG%" >NUL 2>&1
if exist "%OUTPUT_JPG%" (
    echo PASS
    set /a PASS+=1
) else (
    echo FAIL: output file not created
    set /a FAIL+=1
)
:test4skip

REM Test 4: Output is valid JPEG (check first 2 bytes)
set /a TOTAL+=1
echo|set /p="Test 4: Valid JPEG magic bytes ... "
if not exist "%OUTPUT_JPG%" (
    echo FAIL: no output to check
    set /a FAIL+=1
    goto :test5skip
)
REM Use certutil to check hex content
certutil -f -encodehex "%OUTPUT_JPG%" "%TMPDIR%\hex.txt" >NUL 2>&1
for /f "tokens=1,2" %%a in (%TMPDIR%\hex.txt) do (
    if "%%a %%b"=="ff d8" (
        echo PASS
        set /a PASS+=1
    ) else (
        echo FAIL: unexpected magic bytes
        set /a FAIL+=1
    )
    goto :test5skip
)
:test5skip

REM Test 5: Output file size sanity
set /a TOTAL+=1
echo|set /p="Test 5: Output size sanity ....... "
if not exist "%OUTPUT_JPG%" (
    echo FAIL: no output to check
    set /a FAIL+=1
    goto :summary
)
for %%A in ("%OUTPUT_JPG%") do set "FSIZE=%%~zA"
if !FSIZE! GTR 512 (
    if !FSIZE! LSS 1048576 (
        echo PASS ^(!FSIZE! bytes^)
        set /a PASS+=1
    ) else (
        echo FAIL: !FSIZE! bytes too large
        set /a FAIL+=1
    )
) else (
    echo FAIL: !FSIZE! bytes too small
    set /a FAIL+=1
)

:summary
echo.
echo ============================================
echo   Results: %PASS%/%TOTAL% passed, %FAIL% failed
echo ============================================

REM Cleanup
rmdir /s /q "%TMPDIR%" 2>NUL

if %FAIL% GTR 0 exit /b 1
exit /b 0
