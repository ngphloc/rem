@echo off

cd ..\..

call .\build.bat -Dinclude-runtime-lib-src=embed jar-all-src

cd tools\build-tools

@echo on

