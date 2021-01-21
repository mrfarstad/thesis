project_folder=$(echo ${PWD} | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
$project_folder/scripts/run.sh prod hpclab13
