EXPNAME="$1"
PORT=${2:-6006}
EXPROOT=${3:-"/media/hdd5tb/tiam/sandbox/experiments"}

tensorboard --logdir "$EXPROOT"/"$EXPNAME" --port $PORT
