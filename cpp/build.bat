@echo off
REM Configure + build the C++ port with MSVC + Ninja.
REM Usage:  cpp\build.bat  [Release|Debug]
setlocal
set CFG=%1
if "%CFG%"=="" set CFG=Release

set "VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo failed to init MSVC & exit /b 1)

cd /d "%~dp0"
set "PATH=%CD%\..\.venv\Scripts;%PATH%"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=%CFG%
if errorlevel 1 exit /b 1
cmake --build build -j
exit /b %errorlevel%
