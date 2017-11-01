/* DOMAIN */
#define XS                      32
#define YS                      32
#define ZS                      32
#define XWM                     6
#define YWM                     6
#define ZWM                     6
#define XBBM                    1.f
#define YBBM                    1.f
#define ZBBM                    1.f

/* DPD */
#define numberdensity           4
#define kBT                     1.0
#define dt                      0.0005
#define dpd_mass                1.0
#define rbc_mass                1.0
#define solid_mass              1.0

#define adpd_b                  (75.0/numberdensity)    /* solvent */
#define adpd_r                  (75.0/numberdensity)    /* RBC */
#define adpd_br                 (75.0/numberdensity)    /* wall */
#define mygdpd 4.5
#define gdpd_b                  (mygdpd)                /* solvent */
#define gdpd_r                  (mygdpd)                /* RBC */
#define gdpd_br                 (mygdpd)                /* wall */

/* FEATURES */
#define FORCE_CONSTANT
#define FORCE_PAR_A             0.0061
#define solids                  true
#define sbounce_back            true
#define empty_solid_particles   true
#define contactforces           true
#define fsiforces               true
#define ljsigma                 0.3
#define ljepsilon               0.44
#define walls                   true
#define wall_creation           1000
#define tend                    100000

/* FLOW TYPE */
#define pushflow                true
#define pushsolid               true

/* DUMPS */
#define dump_all_fields         false
#define part_freq               50000
#define field_dumps             true
#define field_freq              50000
#define strt_dumps              true
#define strt_freq               500000
