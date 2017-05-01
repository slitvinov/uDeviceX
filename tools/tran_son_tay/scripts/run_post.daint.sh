for d in $(seq 27 117); do
    (
    cd simulations/run_$d
    dt=$(awk '$1 == "dt"             {print $2}' params.txt)
    sh=$(awk '$1 == "_gamma_dot"     {print $2}' params.txt)
    st=$(awk '$1 == "steps_per_dump" {print $2}' params.txt)
    go=$(awk '$1 == "_gammadpd_out"  {print $2}' params.txt)
    gi=$(awk '$1 == "_gammadpd_in"   {print $2}' params.txt)
    gc=$(awk '$1 == "RBCgammaC"      {print $2}' params.txt)
	echo "Processing run $d with parameters $go, $gi, $gc"
    post.py --dt=$dt --ply=ply --sh=$sh --st=$st
    )
done
