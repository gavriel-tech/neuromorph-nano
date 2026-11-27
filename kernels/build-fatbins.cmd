@echo off
rem Rebuilds kernels\build\*.fatbin from kernels\cukern\ (only when a kernel changes; then run
rem tools\gen-integrity.ps1 and neuromorph-nano selftest engine).
rem Visual Studio C++ tools: VCVARS names vcvars64.bat, else vswhere finds the newest install.
rem CUDA toolkit: NVCC names nvcc.exe, else %CUDA_PATH%\bin\nvcc.exe.
setlocal
cd /d "%~dp0.."

if not defined VCVARS (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
)
if not defined VCVARS if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find VC\Auxiliary\Build\vcvars64.bat`) do set "VCVARS=%%i"
)
if not defined VCVARS (
  echo build-fatbins: Visual Studio C++ tools not found; set VCVARS to the path of vcvars64.bat
  exit /b 1
)
if not exist "%VCVARS%" (
  echo build-fatbins: VCVARS=%VCVARS% does not exist
  exit /b 1
)
if not defined NVCC if defined CUDA_PATH set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not defined NVCC (
  echo build-fatbins: CUDA toolkit not found; set CUDA_PATH or NVCC
  exit /b 1
)
if not exist "%NVCC%" (
  echo build-fatbins: NVCC=%NVCC% does not exist
  exit /b 1
)
echo build-fatbins: %VCVARS%
echo build-fatbins: %NVCC%
if "%~1"=="--detect" exit /b 0
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

set "FLAGS=-I"kernels\cbridg\include" -DEMMA_BUILD -DNDEBUG -std=c++20 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90 --default-stream per-thread"
"%NVCC%" -fatbin "kernels\cukern\element.cu" -o "kernels\build\element.fatbin" %FLAGS% --fmad=false -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\linear.cu" -o "kernels\build\linear.fatbin" %FLAGS% --fmad=true -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\attention.cu" -o "kernels\build\attention.fatbin" %FLAGS% --fmad=false -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\prefill.cu" -o "kernels\build\prefill.fatbin" %FLAGS% --fmad=false -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\state.cu" -o "kernels\build\state.fatbin" %FLAGS% --fmad=false -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\segment.cu" -o "kernels\build\segment.fatbin" %FLAGS% --fmad=false -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\flash.cu" -o "kernels\build\flash.fatbin" %FLAGS% --fmad=false -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\q8half.cu" -o "kernels\build\q8half.fatbin" %FLAGS% --fmad=true -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\filter.cu" -o "kernels\build\filter.fatbin" %FLAGS% --fmad=false -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\fast.cu" -o "kernels\build\fast.fatbin" %FLAGS% --fmad=true -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\fastel.cu" -o "kernels\build\fastel.fatbin" %FLAGS% --fmad=false -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
"%NVCC%" -fatbin "kernels\cukern\bmq8.cu" -o "kernels\build\bmq8.fatbin" %FLAGS% --fmad=true -Xcompiler "/utf-8 /EHsc /O2"
if errorlevel 1 exit /b 1
echo build-fatbins: done
endlocal
