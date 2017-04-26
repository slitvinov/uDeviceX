#!/usr/bin/awk -f

function process(f) {
    nr = 0
    while (getline < f > 0) {
	nr++
	if (nr == 1) header($0)
    }
}

function header(l,   key, val, i, n) {
    sub(/^[ \t]*#/, "", l)
    n = split(l, a)
    for (i = 1; i <= n; i++) {
	key = a[i]; val = i
	idx[key] = val
    }
}

BEGIN {
    while (ARGC > 1) {
	process(ARGV[1])
	shift()
    }
}

function shift(  i) {for (i = 1; i  < ARGC; i++) ARGV[i-1] = ARGV[i]; ARGC--}
