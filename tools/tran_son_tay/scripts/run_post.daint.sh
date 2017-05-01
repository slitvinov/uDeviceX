for d in $(seq 2 5); do
    (
    echo "Processing run $d"
    cd simulations/run_$d
    dt=$(awk '$1 == "dt"             {print $2}' params.txt)
    sh=$(awk '$1 == "_gamma_dot"     {print $2}' params.txt)
    st=$(awk '$1 == "steps_per_dump" {print $2}' params.txt)
    python ~/rbc_uq/tools/tran_son_tay/scripts/post.py --dt=$dt --ply=ply --sh=$sh --st=$st
    )
done
