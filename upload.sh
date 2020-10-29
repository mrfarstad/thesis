scp "$1".cu yme:~/thesis/"$1".cu && ssh yme -tt "cd thesis; ./$1.sh"
