#!/bin/bash

# Convert $1 to rational expression
# ./rat       0.2 -> 1/5
# ./rat.sh  -1e-3 -> (-1)/1000

x=$1
ratepsilon=1e-6
tmp=`mktemp /tmp/rat.sh.XXXX`
MAXIMA=${MAXIMA-maxima}

$MAXIMA --very-quiet -r \
'x : '$x'$
tmp: "'$tmp'"$
ratepsilon: '$ratepsilon'$
ratprint: false$
with_stdout(tmp,
printf(true, "~a", rat(x)))$' > /dev/null

cat $tmp
rm $tmp
