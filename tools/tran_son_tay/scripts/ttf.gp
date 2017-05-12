#!/usr/local/bin/gnuplot

reset

D = "data/tab1.txt" # Data
S = "<awk '$1 !~ /#/ && $2 == 15 && $3 == 5 && $4 == 15' data/post.0.txt | sort -g -k 5" # Simulation
R = "data/ttf.0.pdf" # Result

tsc = 13
sz = 0.7
lwd = 2

set terminal pdf size 5*sz, 3*sz linewidth lwd
set output R
set xlabel "shear rate"
set xrange [0:15]
set xtics 2
set ylabel "TTF"
set ytics 0.05
set yrange [0.1:0.3]

plot S  u 5:6                      w lp pt 6 lc "light-red" t "simulation", \
     "" u 5:6:7                    w e  pt 6 lc "light-red" t "", \
     D  u ($1/tsc):($6/$1)         w lp pt 7 lc "dark-red"  t "data", \
     "" u ($1/tsc):($6/$1):($7/$1) w e  pt 7 lc "dark-red"  t ""

set output
