H = "<awk '$1==15 && $2==5 && $3==15' data/post.txt" # sHear
T = "data/stretching.txt" # sTretching
D = "data/tab2.txt" # Data
a1 = 2; t1 = 0.02; f1(x) = a1+(1-a1)*exp(-t1*x); fit f1(x) H u 4:7 via a1, t1
a2 = 2; t2 = 0.02; f2(x) = a2+(1-a2)*exp(-t2*x); fit f2(x) H u 4:9 via a2, t2
tsc = 30; fsc = 25; D0 = 8 # time scale, force scale, initial diameter
plot H u 4:7 w lp lt 1 t "", "" u 4:7:8  w e lt 1 t "shear simulation", \
     H u 4:9 w lp lt 1 t "", "" u 4:9:10 w e lt 1 t "", \
     D u ($1/tsc):($2*cos($5*pi/180)/4) w lp lt 3 t "shear data", \
     D u ($1/tsc):($4/(0.5*D0)) w lp lt 3 t "", \
     T u ($1/fsc):($2/D0) w lp lt 4 t "", "" u ($1/fsc):($2/D0):($3/D0) w e lt 4 t "stretching simulation", \
     T u ($1/fsc):($4/D0) w lp lt 4 t "", "" u ($1/fsc):($2/D0):($5/D0) w e lt 4 t ""
#     f1(x) lt 2 t "shear fit", \
#     f2(x) lt 2 t "", \
