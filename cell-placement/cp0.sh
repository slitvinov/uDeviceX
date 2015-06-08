#!/bin/sh
# A wrapper for cp0.mac
# Usage:
#   ./cp0.sh -r '400 400 40' -i sphere.dat -t 5.0 -o rbcs-ic.txt -c ctcs-ic.txt -f 0.1
#   ./cp0.sh -r '400 400 40' -i sphere.dat -t 5.0 -o r.txt -c c.txt -f 0.1

prog=cp0.sh
base_dir="$(dirname "$0")"
command -v ${MAXIMA:-maxima}  >/dev/null 2>&1 || { echo >&2 "cannot find maxima. set MAXIMA shell variable"; exit 1; }

usage(){
	echo " "
	echo "$prog - generate an initial configuration of RBC"
	echo "Usage:      $prog [-h] [-r 'rank_x rank_y rank_z']"
	echo "            [-t tolerance] [-i RBC template file] [-o RBC output file]"
	echo "            [-c CTC output file] [-f fraction of CTC (0-1)]"
	echo " "
}

# default options
opt_tol=0.1
opt_in="sphere.dat"
opt_orbc="rbcs-ic.txt"
opt_octc="ctcs-ic.txt"
opt_fr=0.25
opt_rank=

while getopts 'r:t:i:o:c:f:h' flag;
do
  case "${flag}" in
      r) opt_rank="${OPTARG}" ;;
      t) opt_tol="${OPTARG}" ;;
      i) opt_in="${OPTARG}" ;;
      o) opt_orbc="${OPTARG}" ;;
      c) opt_octc="${OPTARG}" ;;
      f) opt_fr="${OPTARG}" ;;
      h) usage; exit 1;;
      *) usage; exit 1;;
  esac
done

if [ -z "${opt_rank}" ]; then
    echo " "
    echo "(cp0.sh) (-r 'rank_x rank_y rank_z') rank should be given"
    echo " "
    exit 1
fi

${MAXIMA:-maxima} --very-quiet -r \
		     "rank: \"${opt_rank}\"$ tol: ${opt_tol}$ \
                      out_rbc: \"${opt_orbc}\"$ \
                      out_ctc: \"${opt_octc}\"$ \
                      in_file: \"${opt_in}\"$ \
                      fr     :   ${opt_fr}$ \
                      batchload(\"${base_dir}/cp0.mac\")$"
