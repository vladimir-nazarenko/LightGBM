#!/bin/bash

set -e

mkdir -p hbb_build

docker run -i -t --rm -v `pwd`:/lgbm -u `id -u $USER` phusion/holy-build-box-64 /hbb/activate-exec bash /lgbm/hbb_build.sh

rm -rf hbb_build