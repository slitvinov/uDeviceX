$B/algo/edg/imp.o: $S/utils/error.h $S/algo/edg/imp.h $S/algo/edg/imp/main.h
$B/algo/kahan_sum/imp.o: $S/utils/imp.h $S/utils/error.h $S/algo/kahan_sum/imp.h $S/algo/kahan_sum/imp/main.h $S/algo/kahan_sum/imp/type.h
$B/algo/vectors/imp.o: $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/algo/vectors/imp.h $S/algo/vectors/imp/main.h $S/algo/vectors/imp/type.h $S/math/tform/imp.h $S/coords/imp.h
$B/conf/imp.o: $S/utils/imp.h $S/utils/error.h $S/conf/imp/set.h $S/conf/imp.h $S/conf/imp/main.h $S/conf/imp/type.h $S/conf/imp/get.h $S/utils/msg.h
$B/coords/conf.o: $S/utils/imp.h $S/coords/ini.h $S/utils/error.h $S/conf/imp.h
$B/coords/imp.o: $S/utils/imp.h $S/inc/conf.h $S/coords/ini.h $S/utils/error.h $S/mpi/wrapper.h $S/coords/imp.h $S/utils/mc.h $B/conf.h $S/coords/imp/main.h $S/coords/imp/type.h $S/coords/type.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/io/mesh_read/edg/imp.o: $S/utils/imp.h $S/utils/error.h $S/io/mesh_read/edg/imp/main.h $S/algo/edg/imp.h $S/io/mesh_read/edg/imp/type.h $S/utils/msg.h
$B/io/mesh_read/imp.o: $S/utils/imp.h $S/utils/error.h $S/io/mesh_read/imp/ply.h $S/io/mesh_read/imp.h $S/io/mesh_read/imp/main.h $S/io/mesh_read/imp/type.h $S/io/mesh_read/edg/imp.h $S/io/mesh_read/imp/off.h $S/utils/msg.h
$B/math/tform/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/math/tform/imp.h $B/conf.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/msg.h
$B/math/tri/imp.o: $S/math/tri/imp.h $S/math/tri/dev.h
$B/mesh/area/imp.o: $S/utils/imp.h $S/utils/error.h $S/math/tri/imp.h $S/mesh/area/imp.h $S/algo/kahan_sum/imp.h $S/io/mesh_read/imp.h $S/algo/vectors/imp.h $S/mesh/area/imp/main.h $S/mesh/area/imp/type.h $S/utils/msg.h
$B/mesh/volume/imp.o: $S/utils/imp.h $S/utils/error.h $S/mesh/volume/imp.h $S/algo/kahan_sum/imp.h $S/io/mesh_read/imp.h $S/algo/vectors/imp.h $S/mesh/volume/imp/main.h $S/mesh/volume/imp/type.h $S/utils/msg.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/rbc/gen/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/mpi/wrapper.h $S/rbc/gen/imp.h $S/io/mesh_read/imp.h $S/inc/def.h $S/utils/mc.h $B/conf.h $S/rbc/gen/imp/main.h $S/rbc/matrices/imp.h $S/utils/msg.h $S/coords/imp.h
$B/rbc/matrices/imp.o: $S/utils/imp.h $S/utils/error.h $S/rbc/matrices/imp.h $S/rbc/matrices/imp/main.h $S/rbc/matrices/imp/type.h $S/coords/imp.h $S/utils/msg.h
$B/u/rbc/gen/main.o: $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/rbc/gen/imp.h $S/conf/imp.h $S/mpi/wrapper.h $S/io/mesh_read/imp.h $S/algo/vectors/imp.h $S/mesh/volume/imp.h $S/utils/mc.h $S/rbc/matrices/imp.h $S/mesh/area/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/nvtx/imp.o: $S/utils/error.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
$B/utils/string/imp.o: $S/utils/error.h $S/utils/string/imp.h
