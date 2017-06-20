#!/usr/local/bin/gnuplot

D = "data/tab1.txt" # shear Data
S = "<awk '$1 !~ /#/ && $2 == 15 && $3 == 5 && $4 == 5' data/post.june-19.txt | sort -g -k 5" # shear Simulation
P = "june-19.gc5"

tsc = 35 # time scale
R0 = 4 # initial radius
sz = 0.7
lwd = 2

# Diameters
R = "data/diam.".P.".pdf"
reset
set terminal pdf size 5*sz, 3*sz linewidth lwd
set output R
set xlabel "shear rate"
set xrange [0:200]
set xtics 50
set ylabel "RBC diameters"
set yrange [0:2.5]
set ytics 0.5
set key center
# a1 = 2; t1 = 0.02; f1(x) = a1+(1-a1)*exp(-t1*x); fit f1(x) S u 5:18 via a1, t1
# a2 = 2; t2 = 0.02; f2(x) = a2+(1-a2)*exp(-t2*x); fit f2(x) S u 5:22 via a2, t2
plot S  u (tsc*$5):18       w lp pt 6 lc "light-red"  t "simulation", \
     "" u (tsc*$5):18:19    w e  pt 6 lc "light-red"  t "", \
     "" u (tsc*$5):22       w lp pt 6 lc "light-red"  t "", \
     "" u (tsc*$5):22:23    w e  pt 6 lc "light-red"  t "", \
     D  u 1:($2/R0)         w lp pt 7 lc "dark-red"   t "data", \
     "" u 1:($2/R0):($3/R0) w e  pt 7 lc "dark-red"   t "", \
     "" u 1:($4/R0)         w lp pt 7 lc "dark-red"   t "", \
     "" u 1:($4/R0):($5/R0) w e  pt 7 lc "dark-red"   t ""
set output


# TTF
R = "data/ttf.".P.".pdf" # Result
reset
set terminal pdf size 5*sz, 3*sz linewidth lwd
set output R
set xlabel "shear rate"
set xrange [0:500]
set xtics 50
set ylabel "TTF"
set ytics 0.05
set yrange [0.15:0.3]
plot S  u (tsc*$5):6        w lp pt 6 lc "light-red" t "simulation", \
     "" u (tsc*$5):6:7      w e  pt 6 lc "light-red" t "", \
     D  u 1:($6/$1)         w lp pt 7 lc "dark-red"  t "data", \
     "" u 1:($6/$1):($7/$1) w e  pt 7 lc "dark-red"  t ""
set output
