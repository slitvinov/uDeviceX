d=$1
i=$2
m1=$3 # M for shear 1

cd $d/run_$i

dt=$(awk '$1 == "dt"             {print $2}' params.txt)
sh=$(awk '$1 == "_gamma_dot"     {print $2}' params.txt)
st=$(awk '$1 == "steps_per_dump" {print $2}' params.txt)
go=$(awk '$1 == "_gammadpd_out"  {print $2}' params.txt)
gi=$(awk '$1 == "_gammadpd_in"   {print $2}' params.txt)
gc=$(awk '$1 == "RBCgammaC"      {print $2}' params.txt)
m2=$(awk '$1 != "#" {n+=$6; m+=1} END {print n/m}' diag.txt)

sh=$(awk -v sh=$sh -v m1=$m1 -v m2=$m2 'BEGIN {print m2/m1}')
echo "Processing run $d $i with parameters $go $gi $gc $sh"
post.py --dt=$dt --ply=ply --sh=$sh --st=$st
