/* a hellper class for index shift in an array (see mpi-dpd/main_ring.cpp)

   a circular shift indexing
   Example:
    Ring r(10);
    r.shift(-1);
    r.shift(-1);

    // after two shifts
    // it should look like
    //
    // i:    0 1 2 3 4 5 6 7 8 9
    // r(i): 8 9 0 1 2 3 4 5 6 7
*/

class Ring
{
    const int size;
    int z;
public:
    Ring(const int size): size(size), z(0) {};
    void shift(const int pshift = 1) {
	z += pshift;
	z %= size;
	z = z<0 ? z + size : z;
    }
    int operator()(const int i) {
	const int aux = (z+i)%size;
	return aux<0 ? aux + size : aux;
    }
};
