#!/bin/sh

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

current_location=$(pwd)

docker run --gpus 0 --privileged --rm -it \
           --volume=$XSOCK:$XSOCK:rw \
           --volume=$XAUTH:$XAUTH:rw \
           --volume=$HOME:$HOME \
		   -v /dev:/dev \
		   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
		   -v $current_location/workspace:/home/melodic/workspace \
		   -v $current_location/examples:/home/melodic/examples \
		   -w /home/melodic/workspace \
           --shm-size=1gb \
           --env="XAUTHORITY=${XAUTH}" \
           --env="DISPLAY=${DISPLAY}" \
           --env=TERM=xterm-256color \
           --env=QT_X11_NO_MITSHM=1 \
           --net=host \
           -u "melodic"  \
           carla:0.9.11 \
           bash
