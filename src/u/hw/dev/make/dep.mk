$B/d/api.o: $S/d/cpu/imp.h $S/inc/conf.h $S/d/common.h $S/d/api.h $S/msg.h $B/conf.h $S/d/cuda/imp.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h $S/mpi/glb.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/msg.h $S/mpi/glb.h
$B/u/hw/dev/main.o: $S/inc/conf.h $S/utils/cc.h $S/u/hw/dev/dev.h $S/inc/dev.h $S/msg.h $S/d/api.h $B/conf.h $S/utils/kl.h $S/mpi/glb.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/d/api.h $S/msg.h $B/conf.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/mc.h $B/conf.h
$B/utils/os.o: $S/utils/os.h