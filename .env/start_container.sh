#!/bin/sh

#set variables
UID=1000
GID=1000
UNAME=jupyter
CWD=$(dirname $(realpath "$0"))
WORKBENCH_DIR=/home/$USER/Workbench

#set cwd
cd $CWD
mkdir -p $CWD/home

#build image
docker build -f Dockerfile.dev  \
    -t $USER/co3320-ngc-jupyter \
    --build-arg GID=$GID \
    --build-arg UID=$UID \
    --build-arg UNAME=$UNAME \
    --build-arg GNAME=$UNAME .

#run container
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -it --rm \
    --name co3320-ngc-jupyter \
    --user 1000:1000 \
    -v $WORKBENCH_DIR:/workspace/workbench \
    -v $CWD/home:/home/$UNAME \
    -p 8888:8888 \
    $USER/co3320-ngc-jupyter