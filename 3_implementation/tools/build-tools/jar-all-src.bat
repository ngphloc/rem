@echo off

cd ..\..
if not "%1" == "" goto include-runtime-lib-src
goto none-include-runtime-lib-src

:include-runtime-lib-src
call .\build.bat -Dinclude-runtime-lib-src=%1 jar-all-src
goto end

:none-include-runtime-lib-src
call .\build.bat jar-all-src

:end
cd tools\build-tools

@echo on

