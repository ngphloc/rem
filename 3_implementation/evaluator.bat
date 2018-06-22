@echo off

echo "You should set environmental variable JDK_HOME or JAVA_HOME"

set ANT_HOME=.\tools\ant
echo ANT_HOME = %ANT_HOME%

if "%HUDUP_OLD_PATH%" == "" set HUDUP_OLD_PATH=%PATH%

set PATH=.;%JDK_HOME%\bin;%JAVA_HOME%\bin;%ANT_HOME%\bin;%HUDUP_OLD_PATH%

echo PATH=%PATH%

set CLASSPATH=./hudup.jar;./hudup-server.jar;./hudup-evaluator.jar;./hudup-listener.jar;./hudup-toolkit.jar;hudup-core.jar;./bin;./runtime-lib.jar;./lib/*

echo CLASSPATH=%CLASSPATH%

set JAVA_CMD=java -cp %CLASSPATH%

@echo on

%JAVA_CMD% -Xmx1g net.hudup.Evaluator