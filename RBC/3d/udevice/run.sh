#!/bin/bash


one () {
    printf "processing: %s\n" "$1"
    proof-of-concept/remove_comments.awk      $1                > ${1/.cpp/.s1.cpp}
    proof-of-concept/replace_assignment.awk   ${1/.cpp/.s1.cpp} > ${1/.cpp/.s2.cpp}
    proof-of-concept/replace_memberaccess.awk ${1/.cpp/.s2.cpp} > ${1/.cpp/.s3.cpp}
    proof-of-concept/replace_float.awk        ${1/.cpp/.s3.cpp} > ${1/.cpp/.s4.cpp}
    proof-of-concept/replace_words.awk        ${1/.cpp/.s4.cpp} > ${1/.cpp/.s5.cpp}
    proof-of-concept/replace_ssep.awk         ${1/.cpp/.s5.cpp} > ${1/.cpp/.s6.cpp}
    proof-of-concept/process_fname.awk        ${1/.cpp/.s6.cpp} > ${1/.cpp/.s7.cpp}
    proof-of-concept/rat_float.awk            ${1/.cpp/.s7.cpp} > ${1/.cpp/.s8.cpp}    
    
    cp                                        ${1/.cpp/.s8.cpp}   ${1/.cpp/.mac}
}



rm -rf rbc-cuda

./fsplit.awk data/rbc-cuda.tag.cu
./fsplit.awk data/helper_math.tag.h

for f in rbc-cuda/*.cpp
do
    one $f
done

mkdir -p maxima
cp       rbc-cuda/*.mac maxima/
