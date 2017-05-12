d=$1
sc=$2
for i in $(seq 27 117); do
	run_post.parallel.sh $d $i $sc
done
