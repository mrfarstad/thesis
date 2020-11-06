#!/bin/bash

# Run auto tune framework
python ${PWD}/Autotuning/tuner/tune.py "$1".conf
# Create plt from csv
${PWD}/Autotuning/utilities/output_gnuplot.py results/"$1".csv results/"$1".plt
# Create image from plt
gnuplot -e "set terminal png large size 1500, 1800; set output 'results/$1.png'; load 'results/laplace3d.plt'; exit;"

# If nothing is running on port 8080, then start a local server
# so that you can see the images on the host using
# ssh hpclab13 -N -L localhost:8080:localhost:8080 &
# and heading to localhost:8080
if [ -z "$(lsof -i:8080)" ]; then
    echo "Start localhost:8080 server..."
    python -m SimpleHTTPServer 8080 &> /dev/null &
fi
