cd ../..
if [ "$1" != "" ]
then
	./build.sh -Dinclude-runtime-lib=$1 jar-all
else
	./build.sh jar-all
fi
cd tools/build-tools
