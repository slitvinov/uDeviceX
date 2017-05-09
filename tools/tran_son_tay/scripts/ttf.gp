reset

D = "data/tab1.txt" # Data
S = "<awk 'NR >= 45 && $2 == 15 && $3 == 5 && $4 == 15' data/post.sk.txt | sort -g -k 5" # Simulation
R = "data/ttf.sk.pdf" # Result

tsc = 25
sz = 0.7
lwd = 2

set terminal pdf size 5*sz, 3*sz linewidth lwd
set output R
set xlabel "shear rate"
set xrange [0:9]
set xtics 2
set ylabel "TTF"
set ytics 0.02

plot S u 5:6              w lp pt 6 lc "light-red" t "simulation", \
     D u ($1/tsc):($6/$1) w lp pt 7 lc "dark-red"  t "data"

set output
