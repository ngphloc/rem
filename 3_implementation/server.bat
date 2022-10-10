set EXTRA_CLASSPATH=./hudup-server.jar

call .\env.bat

@echo off

if "%1" == "service" goto service
goto normal

:service
start %JAVAW_CMD% net.hudup.Server %1 %2 %3 %4
goto end

:normal
%JAVA_CMD% net.hudup.Server %1 %2 %3 %4
goto end

:end

@echo on

