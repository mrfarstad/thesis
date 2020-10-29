#! /bin/sh
# rebuild prog if necessary
make simple2DFD
# run prog with some arguments
./simple2DFD "$@"
#cmp --silent coop_snapshots/snap_at_step_"$2" snapshots/snap_at_step_"$2" && echo '### SUCCESS: Files Are Identical! ###' || echo '### WARNING: Files Are Different! ###'
