project_folder=$(echo $(dirname "$0") | sed 's/thesis.*/thesis/')
source $project_folder/constants.sh
$project_folder/autotune.sh hpclab13 $1 stencil > $project_folder/results/out.txt
awk '{if ($1=="rms" && $2=="error") print}' $project_folder/results/out.txt > $project_folder/results/errors.txt
#stdbuf -o 0 -e 0 $project_folder/scripts/autotune.sh hpclab13 $1 stencil | tee $project_folder/scripts/results/out.txt
