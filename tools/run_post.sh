d=simulations

for i in `seq 138 -1 135`; do
    # ii=`printf '%03d\n' $i`
    ii=$i
    dd=$d/run_$ii

    m=`awk -v i=$i 'BEGIN {print i % 4}'`
    # if [[ m -eq 0 ]]; then
    #     sc=`sh get_sc.sh $dd/diag.txt`
    # fi
    gc=`awk '$1 == "RBCgammaC" {print $2}' $dd/params.txt`

    sh run_post.parallel.sh $dd $sc
    res=`awk '$1 == "pa" {a = $2}
              $1 == "pc" {c = $2}
              $1 == "fr" {fr = $2}
              $1 == "sh" {sh = $2}
              $1 == "el" {el = $2}
              END {if (el < 0.05) {print sh, a, c, fr}}' $dd/post.txt`
    echo $gc $res >> $d/res1.txt
done

(
cd $d
fit.py res1.txt | tee res2.txt
)
