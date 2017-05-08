reset

HS = "<awk '$2 == 15 && $3 == 5 && $4 == 15' data/post.new.txt | sort -g -k 5" # sHear Simulation
HD = "data/tab1.txt" # sHear Data

tsc = 25
sz = 0.7
lwd = 2

set terminal pdf size 5*sz, 3*sz linewidth lwd
set output "data/ttf.pdf"
set xlabel "shear rate"
set xrange [0:9]
set xtics 2
set ylabel "TTF"
set ytics 0.02

plot HS u 5:6              w lp pt 6 lc "light-red" t "simulation", \
     HD u ($1/tsc):($6/$1) w lp pt 7 lc "dark-red"  t "data"

set output
