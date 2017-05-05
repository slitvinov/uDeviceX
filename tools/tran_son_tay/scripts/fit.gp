HS = "<awk '$2 == 15 && $3 == 5 && $4 == 15' data/post.new.txt | sort -g" # sHear Simulation
HD = "data/tab2.txt" # sHear Data
TS = "data/str_sim.txt" # sTretching Simulation
TD = "data/str_data.txt" # sTretching Data

set macro
D0a = `awk 'NR==3 {print $2}' @TS`; D0t = `awk 'NR==3 {print $4}' @TS`

# a1 = 2; t1 = 0.02; f1(x) = a1+(1-a1)*exp(-t1*x); fit f1(x) HS u 5:18 via a1, t1
# a2 = 2; t2 = 0.02; f2(x) = a2+(1-a2)*exp(-t2*x); fit f2(x) HS u 5:22 via a2, t2

tsc = 25; fsc = 25; R0 = 4 # time scale, force scale, initial radius
plot HS u 5:18                            w  p lc "light-red"  t "shear simulation", \
     "" u 5:22                            w  p lc "light-red"  t "", \
     "" u 5:18:19                         w e  lc "light-red"  t "", \
     "" u 5:22:23                         w e  lc "light-red"  t "", \
     HD u ($1/tsc):($2*cos($5*pi/180)/R0) w lp lc "dark-red"   t "shear data", \
     "" u ($1/tsc):($4/R0)                w lp lc "dark-red"   t "", \
     TS u ($1/fsc):($2/D0a)               w  p lc "light-blue" t "stretching simulation", \
     "" u ($1/fsc):($4/D0t)               w  p lc "light-blue" t "", \
     "" u ($1/fsc):($2/D0a):($3/D0a)      w e  lc "light-blue" t "", \
     "" u ($1/fsc):($4/D0t):($5/D0t)      w e  lc "light-blue" t "", \
     TD u ($1/fsc):($2/D0a)               w lp lc "dark-blue"  t "stretching data", \
     "" u ($1/fsc):($4/D0t)               w lp lc "dark-blue"  t "", \
     "" u ($1/fsc):($2/D0a):($3/D0a)      w e  lc "dark-blue"  t "", \
     "" u ($1/fsc):($4/D0t):($5/D0t)      w e  lc "dark-blue"  t ""
#     f1(x) lc 2 t "shear fit", \
#     f2(x) lc 2 t "", \
