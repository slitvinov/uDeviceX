#!/usr/bin/env python
import os
import sys

pt = {}  # parameters templates
pv = {}  # parameters values
ps = {}  # parameters short names

# files and directories
home = os.path.expanduser('~')
dpd_dir = home+'/rbc_shear_tags/mpi-dpd'
tools = home+'/rbc_shear_tags/tools'
cnf_file = dpd_dir+'/.conf.h'
rbc_file = dpd_dir+'/params/rbc.inc0.h'
ic_file = 'rbcs-ic.txt'
src_file = 'points.txt'
par_file = 'params.txt'
post_file = 'post.txt'
res_dir = 'simulations'


def gen_templates():
    pt['rc']                   = '#define %s ( %g )\n'
    pt['XS']                   = '#define %s ( %d )\n'
    pt['YS']                   = '#define %s ( %d )\n'
    pt['ZS']                   = '#define %s ( %d )\n'
    pt['XMARGIN_WALL']         = '#define %s ( %d )\n'
    pt['YMARGIN_WALL']         = '#define %s ( %d )\n'
    pt['ZMARGIN_WALL']         = '#define %s ( %d )\n'
    pt['_numberdensity']       = '#define %s ( %d )\n'
    pt['_kBT']                 = '#define %s ( %g )\n'
    pt['dt']                   = '#define %s ( %g )\n'
    pt['rbc_mass']             = '#define %s ( %g )\n'
    pt['_gamma_dot']           = '#define %s ( %g )\n'
    pt['_hydrostatic_a']       = '#define %s ( %g )\n'
    pt['_aij_out']             = '#define %s ( %g )\n'
    pt['_aij_in']              = '#define %s ( %g )\n'
    pt['_aij_rbc']             = '#define %s ( %g )\n'
    pt['_aij_wall']            = '#define %s ( %g )\n'
    pt['_gammadpd_out']        = '#define %s ( %g )\n'
    pt['_gammadpd_in']         = '#define %s ( %g )\n'
    pt['_gammadpd_rbc']        = '#define %s ( %g )\n'
    pt['_gammadpd_wall']       = '#define %s ( %g )\n'
    pt['_ljsigma']             = '#define %s ( %g )\n'
    pt['_ljepsilon']           = '#define %s ( %g )\n'
    pt['RBCx0']                = '#define %s ( %g )\n'
    pt['RBCka']                = '#define %s ( %g )\n'
    pt['RBCkd']                = '#define %s ( %g )\n'
    pt['RBCgammaC']            = '#define %s ( %g )\n'
    pt['RBCmpow']              = '#define %s ( %g )\n'
    pt['RBCphi']               = '#define %s ( %g )\n'
    pt['RBCnv']                = '#define %s ( %d )\n'
    pt['RBCnt']                = '#define %s ( %d )\n'
    pt['RBCkv']                = '#define %s (( %g ) * rc)\n'
    pt['RBCp']                 = '#define %s (( %g ) / rc)\n'
    pt['RBCkb']                = '#define %s (( %g ) / (rc*rc))\n'
    pt['RBCtotArea']           = '#define %s (( %g ) / (rc*rc))\n'
    pt['RBCkbT']               = '#define %s (( %g ) / (rc*rc))\n'
    pt['RBCtotVolume']         = '#define %s (( %g ) / (rc*rc*rc))\n'
    pt['contactforces']        = '#define %s ( %s )\n'
    pt['doublepoiseuille']     = '#define %s ( %s )\n'
    pt['hdf5field_dumps']      = '#define %s ( %s )\n'
    pt['hdf5part_dumps']       = '#define %s ( %s )\n'
    pt['pushtheflow']          = '#define %s ( %s )\n'
    pt['rbcs']                 = '#define %s ( %s )\n'
    pt['steps_per_dump']       = '#define %s ( %d )\n'
    pt['steps_per_hdf5dump']   = '#define %s ( %d )\n'
    pt['tend']                 = '#define %s ( %g )\n'
    pt['wall_creation_stepid'] = '#define %s ( %d )\n'
    pt['walls']                = '#define %s ( %s )\n'


def set_defaults():
    pv['rc']                   = 1.5
    pv['XS']                   = 32
    pv['YS']                   = 16
    pv['ZS']                   = 32
    pv['XMARGIN_WALL']         = 6
    pv['YMARGIN_WALL']         = 6
    pv['ZMARGIN_WALL']         = 6
    pv['_numberdensity']       = 3
    pv['_kBT']                 = 0.1
    pv['dt']                   = 5e-4
    pv['rbc_mass']             = 0.5
    pv['_gamma_dot']           = 5
    pv['_hydrostatic_a']       = 0.05
    pv['_aij_out']             = 4
    pv['_aij_in']              = 4
    pv['_aij_rbc']             = 4
    pv['_aij_wall']            = 4
    pv['_gammadpd_out']        = 8
    pv['_gammadpd_in']         = 8
    pv['_gammadpd_rbc']        = 8
    pv['_gammadpd_wall']       = 8
    pv['_ljsigma']             = 0.3
    pv['_ljepsilon']           = 1.0
    pv['RBCx0']                = 0.45
    pv['RBCp']                 = 0.0039
    pv['RBCka']                = 4900.0
    pv['RBCkb']                = 32
    pv['RBCkd']                = 200
    pv['RBCkv']                = 5000.0
    pv['RBCgammaC']            = 0
    pv['RBCtotArea']           = 124.0
    pv['RBCtotVolume']         = 90.0
    pv['RBCkbT']               = 0.1
    pv['RBCmpow']              = 2.0
    pv['RBCphi']               = 6.97
    pv['RBCnv']                = 498
    pv['RBCnt']                = 992
    pv['contactforces']        = 'false'
    pv['doublepoiseuille']     = 'false'
    pv['hdf5field_dumps']      = 'true'
    pv['hdf5part_dumps']       = 'true'
    pv['pushtheflow']          = 'false'
    pv['rbcs']                 = 'true'
    pv['steps_per_dump']       = 320
    pv['steps_per_hdf5dump']   = 320
    pv['tend']                 = 800
    pv['wall_creation_stepid'] = 100
    pv['walls']                = 'true'


def gen_cnf():
    with open(cnf_file, 'w') as f:
        for key, value in pv.iteritems():
            if 'RBC' not in key:
                f.write(pt[key] % (key, value))


def gen_rbc():
    with open(rbc_file, 'w') as f:
        for key, value in pv.iteritems():
            if 'RBC' in key:
                f.write(pt[key] % (key, value))


def gen_ic(d0):
    with open(d0+'/'+ic_file, 'w') as f:
        ic = '1 0 0 %g  0 1 0 %g  0 0 1 %g  0 0 0 1\n'
        f.write(ic % (pv['XS']/2, pv['YS']/2, pv['ZS']/2))


def gen_dir():
    m = 1; dd = os.listdir(res_dir)
    for d in dd: m = max(m, 1+int(d.split('_')[1]))
    print 'Creating directory #', m
    d0 = '%s/run_%d' % (res_dir, m)
    if not os.path.exists(d0): os.makedirs(d0)
    return d0


def write_par_file(d0):
    with open(d0+'/'+par_file, 'w') as f:
        for key, value in pv.iteritems(): f.write('%s %s\n' % (key, str(value)))


def cp_files(d0):
    cmd = 'cp %s/%s %s/%s'
    os.system(cmd % (dpd_dir, 'test',               d0, ''))
    os.system(cmd % (dpd_dir, 'sdf/wall1/wall.dat', d0, 'sdf.dat'))
    os.system(cmd % (dpd_dir, 'rbc.dat',            d0, ''))


def gen_par(pn0, pv0):
    set_defaults()
    for j in range(len(pv0)): pv[pn0[j]] = float(pv0[j])
    sh = pv['_gamma_dot']
    st = int(1600/sh); pv['steps_per_dump'] = pv['steps_per_hdf5dump'] = st
    te = 80/sh; pv['tend'] = te


def recompile():
    cmd = 'cd %s && make clean && make -j'
    os.system(cmd % dpd_dir)


def pre():
    d0 = gen_dir()
    write_par_file(d0)
    cp_files(d0)
    gen_ic(d0)
    return d0


def run(d0):
    os.system('cd %s && ./test' % d0)
    os.system('cd %s/ply && sh ~/scripts/ply/cm.sh' % d0)


def post(d0):
    sys.path.append(dpd_dir)
    import compute_freq_standalone_Tran_Son_Tay as cT
    import numpy as np

    time, theta, omega = cT.read_data(d0+'/ply', pv['dt'], pv['steps_per_dump'])
    try:
        fr, fru = cT.get_fr(time, omega)
        an, anu = cT.get_an(theta)
        print fr, fru, an, anu

        with open(d0+'/'+post_file, 'w') as f:
            sh = pv['_gamma_dot']
            f.write('freq/shrate: %g %g\n' % (2.*np.pi*fr/sh, fru/sh))
            f.write('angle: %g %g\n' % (an, anu))

        with open(post_file, 'a') as f:
            for t in pn: f.write('%g ' % pv[t])
            sh = pv['_gamma_dot']
            f.write('%g %g ' % (2.*np.pi*fr/sh, fru/sh))
            f.write('%g %g\n' % (an, anu))
    except:
        print 'Unexpected error:', sys.exc_info()[0]
        pass

    os.system('mv Tran-Son-Tay* %s' % d0)


if __name__ == '__main__':
    if not os.path.exists(res_dir): os.makedirs(res_dir)

    gen_templates()

    with open(src_file, 'r') as f: lines = f.readlines()
    pn0 = lines[0].replace('#', '').split()  # user-defined parameters names

    for l in lines:
        pv0 = l.split()  # user-defined parameters values
        if '#' in pv0[0]: continue  # skip comments

        print 'Running line %s\n' % l

        gen_par(pn0, pv0)
        gen_cnf()
        gen_rbc()
        recompile()

        d0 = pre()
        run(d0)
        post(d0)
