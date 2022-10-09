@echo off

cd ..\..

call .\build.bat -Dinclude-runtime-lib=embed jar-all

cd tools\build-tools

@echo on

