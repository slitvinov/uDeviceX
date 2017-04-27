Experimental and analytic results for RBC in shear flow
=======================================================

```
load("dimension");
alias(di, dimension);
lput(L, d):=for e in L do put(e, d, di);

lput(['a, 'b, 'c], 'length);
lput(['la], 'length^2);
lput(['f, 'gd] , 1/'time);
lput(['V], 'length^3);
lput(['th], 1);
Velocity: 'length/'time;
Acceleration: Velocity/'time;
Force : Acceleration * 'mass;
Area  : 'length^2;
Pa: Force / Area;
lput(['eta0], Pa * 'time);
Energy: 'mass * Velocity^2;
Dissipation: Energy/'time;

```

```
0.05386154615194705
```



