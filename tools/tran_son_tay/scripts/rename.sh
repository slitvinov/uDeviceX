#!/bin/sh

ARGS=("$@")
NARGS=$#

lwd=links
echo mkdir -p $lwd

gen_dir () {
    awk -v lwd="$1" -v name="$2" -v par="$3" '
        BEGIN {
            split(name, pn)
            split(par, pv)
            for (j = 1; j in pv; ) {
                printf("mkdir -p %s/", lwd)
                for (i in pn) printf("%s_%s/", pn[i], pv[j++])
                printf("\n")
            }
        } '
}

gen_links () {
    awk -v lwd="$1" -v name="$2" -v par="$3" -v cwd="$4" '
        BEGIN {
            split(name, pn)
            split(par, pv)
            for (j = 1; j in pv; ) {
                rwd = pv[j++]
                printf("ln -s %s/%s %s/", cwd, rwd, lwd)
                for (i in pn) printf("%s_%s/", pn[i], pv[j++])
                sh = pv[j++]
                printf("sh_%.1f\n", sh)
            }
        } '
}

for i in `seq 1 $NARGS`; do
    ii=`awk -v i=$i 'BEGIN {print i-1}'`
    name="$name ${ARGS[$ii]}"
    par=`list "$name" run_*`
    gen_dir "$lwd" "$name" "$par"
done

cwd=`pwd`
par=`list d "$ARGS" sh run_*`
gen_links "$lwd" "$ARGS" "$par" "$cwd"
