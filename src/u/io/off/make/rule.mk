$B/conf/imp.o: $S/conf/imp.cpp; $X -I$S/. -I$S/conf
$B/coords/conf.o: $S/coords/conf.cpp; $X -I$S/. -I$S/coords
$B/coords/imp.o: $S/coords/imp.cpp; $X -I$S/. -I$S/coords
$B/d/api.o: $S/d/api.cu; $N -I$S/. -I$S/d
$B/io/mesh_read/imp.o: $S/io/mesh_read/imp.cpp; $X -I$S/. -I$S/io/mesh_read
$B/mpi/glb.o: $S/mpi/glb.cpp; $X -I$S/. -I$S/mpi
$B/mpi/type.o: $S/mpi/type.cpp; $X -I$S/. -I$S/mpi
$B/mpi/wrapper.o: $S/mpi/wrapper.cpp; $X -I$S/. -I$S/mpi
$B/u/io/off/main.o: $S/u/io/off/main.cpp; $X -I$S/.
$B/utils/cc.o: $S/utils/cc.cpp; $X -I$S/. -I$S/utils
$B/utils/error.o: $S/utils/error.cpp; $X -I$S/. -I$S/utils
$B/utils/imp.o: $S/utils/imp.cpp; $X -I$S/. -I$S/utils
$B/utils/mc.o: $S/utils/mc.cpp; $X -I$S/. -I$S/utils
$B/utils/msg.o: $S/utils/msg.cpp; $X -I$S/. -I$S/utils
$B/utils/os.o: $S/utils/os.cpp; $X -I$S/. -I$S/utils
