CXXFLAGS += ${HDF5_CXXFLAGS}
CXXFLAGS += ${MPI_CXXFLAGS}
CXXFLAGS += ${LIBCONFIG_CXXFLAGS}
NVCCLIBS  = --linker-options '${HDF5_LIBS}     ${MPI_LIBS}     ${LIBCONFIG_LIBS}'
LIBS = ${HDF5_LIBS}  ${MPI_LIBS}  ${NVCC_LIBS} ${LIBCONFIG_LIBS}