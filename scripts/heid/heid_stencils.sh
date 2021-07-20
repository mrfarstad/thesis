#!/bin/bash
project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
bash $project_folder/scripts/heid/heid_stencils_heuristic.sh &
bash $project_folder/scripts/heid/heid_stencils_autotuned.sh &
wait
echo "Done!"
#python3 $project_folder/scripts/migrate_stencils_json.py
