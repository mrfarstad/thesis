project_folder=$(echo $(dirname "$0") | sed 's/thesis.*/thesis/')
make -C $project_folder ID=profile BLOCK_X=$1 BLOCK_Y=$2
make -C $project_folder profile ID=profile
