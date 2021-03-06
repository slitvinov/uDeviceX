struct GenColor;
struct Particle;
struct Coords;
struct Config;

void inter_color_ini(GenColor**);
void inter_color_fin(GenColor*);

void inter_color_set_drop(float R, GenColor*);
void inter_color_set_uniform(GenColor*);

void inter_color_set_conf(const Config*, GenColor*);

void inter_color_apply_hst(const Coords*, const GenColor*, int n, const Particle *pp, /**/ int *cc);
void inter_color_apply_dev(const Coords*, const GenColor*, int n, const Particle *pp, /**/ int *cc);
