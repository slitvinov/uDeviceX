$B/algo/scan/imp.o: $S/algo/scan/cpu/imp.h $S/algo/scan/cuda/imp.h $S/algo/scan/dev.h $S/algo/scan/imp.h $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/msg.h $S/utils/cc.h $S/utils/kl.h
$B/clist/imp.o: $S/algo/scan/imp.h $S/clist/code.h $S/clist/dev.h $S/clist/imp.h $S/clist/imp/fin.h $S/clist/imp/ini.h $S/clist/imp/main.h $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/msg.h $S/partlist/dev.h $S/partlist/imp.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/imp.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/u/clist/main.o: $S/algo/scan/imp.h $S/clist/imp.h $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/dev.h $S/inc/type.h $S/msg.h $S/partlist/imp.h $S/utils/cc.h $S/utils/error.h $S/utils/imp.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/error.h $S/utils/mc.h
$B/utils/os.o: $S/msg.h $S/utils/error.h $S/utils/os.h
