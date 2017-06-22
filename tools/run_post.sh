d=grid/links.daint

for f in `find -L $d -mindepth 9 -maxdepth 9 -name 'params.txt' | sort`; do
    dd=`dirname "$f"`

    if [[ -f $dd/post.txt ]]; then
        continue
    fi

    ddd=`dirname $dd.txt`
    sc=`get_sc.sh $ddd/sh_1.0/diag.txt`
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

# (
# cd $d
# fit.py res1.txt | tee res2.txt
# )
