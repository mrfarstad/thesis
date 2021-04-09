#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
bash $project_folder/scripts/heid/heid_stencil_depths_heuristic.sh &
bash $project_folder/scripts/heid/heid_stencil_depths_autotuned.sh &
wait
echo "Done!"
#python3 $project_folder/scripts/migrate_stencil_depths_json.py
