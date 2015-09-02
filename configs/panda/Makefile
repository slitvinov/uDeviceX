h5part = 1
NVCC = nvcc -I/home/lisergey/prefix-h5part/include/ \
     -I/opt/mpich/include/ -L/opt/mpich/lib/ -I/opt/hdf5_mpich/include/ -L/opt/hdf5_mpich/lib/ \
     -L/home/lisergey/prefix-h5part/lib
CXX = /opt/mpich/bin/mpicxx -I /home/lisergey/prefix-h5part/include/ \
    -L/opt/mpich/lib/ -L/opt/hdf5_mpich/lib/ -O0 -g -L/home/lisergey/prefix-h5part/lib
NVCC_LINK_FLAGS=-L/usr/local/cuda-6.5/bin/..//lib64 \
	-L/usr/local/cuda-6.5/bin/..//lib64/stubs \
	-Wl,-rpath=/usr/local/cuda-6.5/bin/..//lib64
