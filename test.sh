#! /bin/sh
# rebuild prog if necessary
make simple2DFD
# run prog with some arguments
./simple2DFD 2 400  > /dev/null &
./simple2DFD 2 800  > /dev/null &
./simple2DFD 2 1200 > /dev/null &
./simple2DFD 2 1500 > /dev/null &
wait
cmp --silent coop_snapshots/snap_at_step_400 snapshots/snap_at_step_400 &&
cmp --silent coop_snapshots/snap_at_step_800 snapshots/snap_at_step_800 &&
cmp --silent coop_snapshots/snap_at_step_1200 snapshots/snap_at_step_1200 &&
cmp --silent coop_snapshots/snap_at_step_1500 snapshots/snap_at_step_1500 &&
echo '### SUCCESS: Files Are Identical! ###' || echo '### WARNING: Files Are Different! ###'
make clean
