namespace mesh {    
void get_bbox(const float *rr, const int n, /**/ float3 *minbb, float3 *maxbb);

void get_bboxes_hst(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb);
void get_bboxes_dev(const Particle *pp, const int nps, const int ns, /**/ float3 *minbb, float3 *maxbb);
}
