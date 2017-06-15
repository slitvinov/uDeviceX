namespace k_rbc {
#define cross(a, b) make_float3                 \
    ((a).y*(b).z - (a).z*(b).y,                 \
     (a).z*(b).x - (a).x*(b).z,                 \
     (a).x*(b).y - (a).y*(b).x)

__DF__ float3 angle(float3 v1, float3 v2,
		     float3 v3, float area,
		     float volume) {
#include "params/rbc.inc0.h"
    float Ak, A0, n_2, coefArea, coeffVol,
	r, xx, IbforceI_wcl, kp, IbforceI_pow, ka0, kv0, x0, l0, lmax,
	kbToverp;

    float3 x21 = v2 - v1, x32 = v3 - v2, x31 = v3 - v1;
    float3 nn = cross(x21, x31); /* normal */

    Ak = 0.5 * sqrtf(dot(nn, nn));

    A0 = RBCtotArea / (2.0 * RBCnv - 4.);
    n_2 = 1.0 / Ak;
    ka0 = RBCka / RBCtotArea;
    coefArea =
	-0.25f * (ka0 * (area - RBCtotArea) * n_2) -
	RBCkd * (Ak - A0) / (4. * A0 * Ak);

    kv0 = RBCkv / (6.0 * RBCtotVolume);
    coeffVol = kv0 * (volume - RBCtotVolume);
    float3 addFArea = coefArea * cross(nn, x32);
    float3 addFVolume = coeffVol * cross(v3, v2);

    r = sqrtf(dot(x21, x21));
    r = r < 0.0001f ? 0.0001f : r;
    l0 = sqrt(A0 * 4.0 / sqrt(3.0));
    lmax = l0 / RBCx0;
    xx = r / lmax;

    kbToverp = RBCkbT / RBCp;
    IbforceI_wcl =
	    kbToverp * (0.25f / ((1.0f - xx) * (1.0f - xx)) - 0.25f + xx) /
	    r;

    x0 = RBCx0;
    kp =
	    (RBCkbT * x0 * (4 * x0 * x0 - 9 * x0 + 6) * l0 * l0) /
	    (4 * RBCp * (x0 - 1) * (x0 - 1));
    IbforceI_pow = -kp / powf(r, RBCmpow) / r;

    return addFArea + addFVolume + (IbforceI_wcl + IbforceI_pow) * x21;
}

__DF__ float3 visc(float3 v1, float3 v2,
					 float3 u1, float3 u2) {
    float3 du = u2 - u1, dr = v1 - v2;
    float gammaC = RBCgammaC, gammaT = 3.0 * RBCgammaC;

    return gammaT                             * du +
	   gammaC * dot(du, dr) / dot(dr, dr) * dr;
}

} /* namespace k_rbc */
