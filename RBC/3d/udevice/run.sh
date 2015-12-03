#!/bin/bash

./fsplit.awk data/rbc-cuda.cu

one () {
    proof-of-concept/remove_comments.awk      $1                > ${1/.cpp/.s1.cpp}
    proof-of-concept/replace_assignment.awk   ${1/.cpp/.s1.cpp} > ${1/.cpp/.s2.cpp}
    proof-of-concept/replace_memberaccess.awk ${1/.cpp/.s2.cpp} > ${1/.cpp/.s3.cpp}
    proof-of-concept/replace_float.awk        ${1/.cpp/.s3.cpp} > ${1/.cpp/.s4.cpp}
    proof-of-concept/replace_words.awk        ${1/.cpp/.s4.cpp} > ${1/.cpp/.s5.cpp}
    proof-of-concept/process_fname.awk        ${1/.cpp/.s5.cpp} > ${1/.cpp/.s6.cpp}
    proof-of-concept/replace_ssep.awk         ${1/.cpp/.s6.cpp} > ${1/.cpp/.s7.cpp}

    cp                                        ${1/.cpp/.s7.cpp}   ${1/.cpp/.mac}
}

for f in rbc-cuda/*.cpp
do
    one $f
done
