#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
# Run auto tune framework
python $project_folder/Autotuning/tuner/tune.py $project_folder/configs/$1/$2.conf
#python ${PWD}/Autotuning/tuner/tune.py configs/$1/$2.conf
#python ${PWD}/Autotuning/tuner/tune.py configs/$1.conf
#python ${PWD}/Autotuning/tuner/tune.py configs/hpclab13/base.conf
# Create plt from csv
$project_folder/Autotuning/utilities/output_gnuplot.py $project_folder/results/laplace3d.csv results/laplace3d.plt
# Create image from plt
nuplot -e "set terminal png large size 1500, 1800; set output '$project_folder/results/laplace3d.png'; load '$project_folder/results/laplace3d.plt'; exit;"

# If nothing is running on port 8080, then start a local server
# so that you can see the images on the host using
# ssh hpclab13 -N -L localhost:8080:localhost:8080 &
# and heading to localhost:8080
if [ -z "$(lsof -i:8080)" ]; then
    echo "Start localhost:8080 server..."
    python -m SimpleHTTPServer 8080 &> /dev/null &
fi
