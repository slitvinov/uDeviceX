#define XWM      6
#define YWM      6
#define ZWM      6

#define XBBM 1.f
#define YBBM 1.f
#define ZBBM 1.f

#define numberdensity       4
#define kBT                 4.44302e-8
#define dt                  5e-4
#define rbc_mass            0.5
#define solid_mass          0.5

#define gamma_dot           0.f        /* shear rate */

#define aij_solv            5.f
#define aij_rbc             5.f
#define aij_solid           5.f
#define aij_wall            5.f
#define gammadpd_solv       8.f
#define gammadpd_rbc        8.f
#define gammadpd_solid      8.f
#define gammadpd_wall       8.f

#define ljsigma          0.3f       /* RBC-RBC contact LJ interaction parameters */
#define ljepsilon        0.44f

#define contactforces    (false)
#define doublepoiseuille (false)
#define       strt_dumps  (true)
#define        strt_freq  (5000)
#define      field_dumps (false)
#define       field_freq  (2000)
#define       part_dumps (false)
#define        part_freq  (1000)
#define         pushflow (false)
#define             tend    (50)
#define    wall_creation  (5000)
#define            walls (false)
#define            RBCnv   (498)
