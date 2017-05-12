#!/usr/local/bin/gnuplot

reset

TD = "data/str_data.txt" # sTretching Data
TS = "data/str_sim.txt" # sTretching Simulation
HD = "data/tab1.txt" # sHear Data
HS = "<awk '$1 !~ /#/ && $2 == 15 && $3 == 5 && $4 == 15' data/post.0.txt | sort -g -k 5" # sHear Simulation
R = "data/diam.0.pdf"

set macro
D0a = `awk 'NR==3 {print $2}' @TS`; D0t = `awk 'NR==3 {print $4}' @TS`

tsc = 13 # time scale
fsc = 25 # force scale
R0 = 4 # initial radius
sz = 0.7
lwd = 2

set terminal pdf size 5*sz, 3*sz linewidth lwd
set output R
set xlabel "shear rate"
set xrange [0:15]
set xtics 2
set ylabel "RBC diameters"
set yrange [0:2.5]
set ytics 0.5
set key center

# a1 = 2; t1 = 0.02; f1(x) = a1+(1-a1)*exp(-t1*x); fit f1(x) HS u 5:18 via a1, t1
# a2 = 2; t2 = 0.02; f2(x) = a2+(1-a2)*exp(-t2*x); fit f2(x) HS u 5:22 via a2, t2

plot HS u 5:18                            w lp pt 6 lc "light-red"  t "simulation", \
     "" u 5:18:19                         w e  pt 6 lc "light-red"  t "", \
     "" u 5:22                            w lp pt 6 lc "light-red"  t "", \
     "" u 5:22:23                         w e  pt 6 lc "light-red"  t "", \
     HD u ($1/tsc):($2/R0)                w lp pt 7 lc "dark-red"   t "data", \
     "" u ($1/tsc):($2/R0):($3/R0)        w e  pt 7 lc "dark-red"   t "", \
     "" u ($1/tsc):($4/R0)                w lp pt 7 lc "dark-red"   t "", \
     "" u ($1/tsc):($4/R0):($5/R0)        w e  pt 7 lc "dark-red"   t "", \
#     TS u ($1/fsc):($2/D0a)               w lp lc "cyan"       t "stretching simulation", \
#     "" u ($1/fsc):($2/D0a):($3/D0a)      w e  lc "cyan"       t "", \
#     "" u ($1/fsc):($4/D0t)               w lp lc "cyan"       t "", \
#     "" u ($1/fsc):($4/D0t):($5/D0t)      w e  lc "cyan"       t "", \
#     TD u ($1/fsc):($2/D0a)               w lp lc "dark-blue"  t "stretching data", \
#     "" u ($1/fsc):($2/D0a):($3/D0a)      w e  lc "dark-blue"  t "", \
#     "" u ($1/fsc):($4/D0t)               w lp lc "dark-blue"  t "", \
#     "" u ($1/fsc):($4/D0t):($5/D0t)      w e  lc "dark-blue"  t "", \
#     f1(x) lc 2 t "shear fit", \
#     f2(x) lc 2 t "", \

set output
