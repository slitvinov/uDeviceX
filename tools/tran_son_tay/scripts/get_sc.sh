awk 'NR>1 {n+=$6; m+=1} END {print n/m}' $1
