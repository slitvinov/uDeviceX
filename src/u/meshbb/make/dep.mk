$B/algo/scan/imp.o: $S/algo/scan/cpu/imp.h $S/algo/scan/cuda/imp.h $S/algo/scan/dev.h $S/algo/scan/imp.h $B/conf.h $S/d/api.h $S/d/ker.h $S/d/q.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/msg.h $S/utils/cc.h $S/utils/kl.h
$B/clist/imp.o: $S/algo/scan/imp.h $S/clist/code.h $S/clist/dev.h $S/clist/imp.h $S/clist/imp/fin.h $S/clist/imp/ini.h $S/clist/imp/main.h $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/msg.h $S/partlist/dev.h $S/partlist/imp.h $S/utils/cc.h $S/utils/kl.h
$B/d/api.o: $B/conf.h $S/d/api.h $S/d/common.h $S/d/cpu/imp.h $S/d/cuda/imp.h $S/inc/conf.h $S/utils/error.h $S/utils/halloc.h
$B/glb/imp.o: $B/conf.h $S/d/api.h $S/glb/get.h $S/glb/imp/dec.h $S/glb/imp/main.h $S/glb/imp/util.h $S/glb/set.h $S/glb/wvel/imp.h $S/inc/conf.h $S/mpi/glb.h
$B/glb/wvel/imp.o: $B/conf.h $S/glb/wvel/imp.h $S/glb/wvel/imp/flat.h $S/glb/wvel/imp/sin.h $S/inc/conf.h $S/msg.h
$B/io/off.o: $S/io/off.h $S/io/off/imp.h $S/utils/efopen.h $S/utils/error.h
$B/meshbb/imp.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/dev.h $S/inc/type.h $S/math/dev.h $S/meshbb/bbstates.h $S/meshbb/dev/collect.h $S/meshbb/dev/cubic_root/log_root0.h $S/meshbb/dev/cubic_root/log_root1.h $S/meshbb/dev/cubic_root/main.h $S/meshbb/dev/intersection.h $S/meshbb/dev/main.h $S/meshbb/dev/roots.h $S/meshbb/dev/utils.h $S/meshbb/imp.h $S/meshbb/imp/find_collisions/log_root0.h $S/meshbb/imp/find_collisions/log_root1.h $S/meshbb/imp/main.h $S/meshbb/type.h $S/msg.h $S/utils/cc.h $S/utils/kl.h
$B/mpi/glb.o: $B/conf.h $S/inc/conf.h $S/mpi/glb.h $S/mpi/wrapper.h $S/utils/mc.h
$B/mpi/type.o: $S/inc/conf.h $S/inc/type.h $S/mpi/type.h $S/utils/mc.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/msg.o: $S/mpi/glb.h $S/msg.h
$B/u/meshbb/main.o: $S/algo/scan/imp.h $S/clist/imp.h $B/conf.h $S/d/api.h $S/inc/conf.h $S/inc/def.h $S/inc/dev.h $S/inc/type.h $S/io/off.h $S/meshbb/imp.h $S/mpi/glb.h $S/msg.h $S/partlist/imp.h $S/u/meshbb/imp/main.h $S/utils/cc.h $S/utils/error.h $S/utils/kl.h
$B/utils/cc.o: $B/conf.h $S/d/api.h $S/inc/conf.h $S/utils/cc/common.h $S/utils/error.h
$B/utils/efopen.o: $S/utils/efopen.h $S/utils/error.h
$B/utils/error.o: $S/msg.h $S/utils/error.h
$B/utils/halloc.o: $S/utils/error.h $S/utils/halloc.h
$B/utils/mc.o: $B/conf.h $S/inc/conf.h $S/utils/error.h $S/utils/mc.h
$B/utils/os.o: $S/msg.h $S/utils/error.h $S/utils/os.h
