d=grid
for i in $(seq 28 -1 1); do
	m=$(awk -v i=$i 'BEGIN {print i % 4}')
	if [[ m -eq 0 ]]; then
		sc=$(sh get_sc.sh $d/run_$i/diag.txt)
	fi
	sh run_post.parallel.sh $d $i $sc
done
