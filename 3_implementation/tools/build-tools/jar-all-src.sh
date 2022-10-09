cd ../..
if [ "$1" != "" ]
then
	./build.sh -Dinclude-runtime-lib-src=$1 jar-all-src
else
	./build.sh jar-all-src
fi
cd tools/build-tools
