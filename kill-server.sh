kill $(lsof -i:8080 | tail -n+2 | awk {'print $2'})
