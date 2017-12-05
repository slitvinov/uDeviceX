static __device__ float3 transform(const Particle p) {
    enum {X, Y, Z};
    float3 u; // radial coordinates
    float x, y, r, rinv;
    float cost, sint;
    x = p.r[X];
    y = p.r[Y];

    r = sqrt(x*x + y*y);
    rinv = 1 / r;
    cost = rinv * x;
    sint = rinv * y;
    
    u.x =   cost * p.v[X] + sint * p.v[Y];
    u.y = - sint * p.v[X] + cost * p.v[Y];
    u.z = p.v[Z];
    return u;
}
