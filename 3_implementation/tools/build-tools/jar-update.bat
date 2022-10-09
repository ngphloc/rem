@echo off

cd ..\..
call .\build.bat -Dpackage=%1 jar-update
cd tools\build-tools

@echo on
