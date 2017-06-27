#!/usr/bin/env python
import os
import sys
from math import acos, sqrt, pi
from argparse import ArgumentParser

pt = {}  # parameters templates
pv = {}  # parameters values

# files and directories
home = os.path.expanduser('~')
dpd_dir = home+'/rbc_uq/mpi-dpd'
tools = home+'/rbc_uq/tools'
cnf_file = dpd_dir+'/.conf.h'
rbc_file = dpd_dir+'/params/rbc.inc0.h'
ic_file = 'rbcs-ic.txt'
src_file = 'points.txt'
par_file = 'params.txt'
res_dir = 'grid'


def gen_templates():
    pt['rc']                   = '#define %s ( %g )\n'
    pt['XS']                   = '#define %s ( %d )\n'
    pt['YS']                   = '#define %s ( %d )\n'
    pt['ZS']                   = '#define %s ( %d )\n'
    pt['XMARGIN_WALL']         = '#define %s ( %d )\n'
    pt['YMARGIN_WALL']         = '#define %s ( %d )\n'
    pt['ZMARGIN_WALL']         = '#define %s ( %d )\n'
    pt['numberdensity']        = '#define %s (( %d ) * (rc*rc*rc))\n'
    pt['kBT']                  = '#define %s (( %g ) / (rc*rc))\n'
    pt['dt']                   = '#define %s ( %g )\n'
    pt['rbc_mass']             = '#define %s ( %g )\n'
    pt['gamma_dot']            = '#define %s ( %g )\n'
    pt['hydrostatic_a']        = '#define %s (( %g ) / rc)\n'
    pt['aij_out']              = '#define %s (( %g ) / rc)\n'
    pt['aij_in']               = '#define %s (( %g ) / rc)\n'
    pt['aij_rbc']              = '#define %s (( %g ) / rc)\n'
    pt['aij_wall']             = '#define %s (( %g ) / rc)\n'
    pt['aij_in_out']           = '#define %s (( %g ) / rc)\n'
    pt['gammadpd_out']         = '#define %s ( %g )\n'
    pt['gammadpd_in']          = '#define %s ( %g )\n'
    pt['gammadpd_rbc']         = '#define %s ( %g )\n'
    pt['gammadpd_wall']        = '#define %s ( %g )\n'
    pt['gammadpd_in_out']      = '#define %s ( %g )\n'
    pt['ljsigma']              = '#define %s ( %g )\n'
    pt['ljepsilon']            = '#define %s (( %g ) / (rc*rc))\n'
    pt['RBCrc']                = '#define %s ( %g )\n'
    pt['RBCx0']                = '#define %s ( %g )\n'
    pt['RBCka']                = '#define %s ( %g )\n'
    pt['RBCkd']                = '#define %s ( %g )\n'
    pt['RBCgammaC']            = '#define %s ( %g )\n'
    pt['RBCgammaT']            = '#define %s ( %g )\n'
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
    pt['RBCrnd']               = '#define %s ( %d )\n'
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
    pt['stretchingForce']      = '#define %s (( %g ) / rc)\n'
    pt['strVerts']             = '#define %s ( %d )\n'
    pt['nosolvent']            = '#define %s ( %s )\n'


def set_defaults():
    pv['rc']                   = 1.5
    pv['XS']                   = 32
    pv['YS']                   = 8
    pv['ZS']                   = 32
    pv['XMARGIN_WALL']         = 6
    pv['YMARGIN_WALL']         = 6
    pv['ZMARGIN_WALL']         = 6
    pv['numberdensity']        = 3
    pv['kBT']                  = 0.1
    pv['dt']                   = 8e-4
    pv['rbc_mass']             = 0.5
    pv['gamma_dot']            = 0
    pv['hydrostatic_a']        = 0.05
    pv['aij_out']              = 4
    pv['aij_in']               = 4
    pv['aij_rbc']              = 4
    pv['aij_wall']             = 4
    pv['aij_in_out']           = 4
    pv['gammadpd_out']         = 15
    pv['gammadpd_in']          = 5
    pv['gammadpd_rbc']         = 15
    pv['gammadpd_wall']        = 15
    pv['gammadpd_in_out']      = 10
    pv['ljsigma']              = 0.3
    pv['ljepsilon']            = 1.0
    pv['RBCrc']                = 1
    pv['RBCx0']                = 0.45
    pv['RBCp']                 = 0.0039
    pv['RBCka']                = 4900
    pv['RBCkb']                = 100
    pv['RBCkd']                = 5000
    pv['RBCkv']                = 5000
    pv['RBCgammaC']            = 10
    pv['RBCgammaT']            = 0
    # pv['RBCgammaT']            = 3*pv['RBCgammaC']
    pv['RBCtotArea']           = 124.0
    pv['RBCtotVolume']         = 90.0
    pv['RBCkbT']               = 0.1
    pv['RBCmpow']              = 2.0
    pv['RBCphi']               = 6.97
    pv['RBCnv']                = 498
    pv['RBCnt']                = 992
    pv['RBCrnd']               = 1
    pv['contactforces']        = 0
    pv['doublepoiseuille']     = 0
    pv['hdf5field_dumps']      = 0
    pv['hdf5part_dumps']       = 1
    pv['pushtheflow']          = 0
    pv['rbcs']                 = 1
    pv['steps_per_dump']       = 320
    pv['steps_per_hdf5dump']   = 320
    pv['tend']                 = 800
    pv['wall_creation_stepid'] = 100
    pv['walls']                = 1
    pv['stretchingForce']      = 0
    pv['strVerts']             = 10
    pv['nosolvent']            = 0


def gen_cnf():
    with open(cnf_file, 'w') as f:
        for key, value in sorted(pv.iteritems()):
            if 'RBC' not in key:
                f.write(pt[key] % (key, value))


def gen_rbc():
    with open(rbc_file, 'w') as f:
        for key, value in sorted(pv.iteritems()):
            if 'RBC' in key:
                f.write(pt[key] % (key, value))


def gen_ic(d0):
    with open(d0+'/'+ic_file, 'w') as f:
        sc = pv['RBCrc']
        f.write('%g 0 0 %d  0 %g 0 %d  0 0 %g %d  0 0 0 1\n' % (
                 sc, pv['XS']/2, sc, pv['YS']/2, sc, pv['ZS']/2))


def gen_dir():
    m = 1; dd = os.listdir(res_dir)
    for d in dd:
        w = d.split('_')
        if (w[0] == 'run'): m = max(m, 1+int(w[1]))
    print 'Creating directory #', m
    d0 = '%s/run_%03d' % (res_dir, m)
    if not os.path.exists(d0): os.makedirs(d0)
    return d0


def write_par_file(d0):
    with open(d0+'/'+par_file, 'w') as f:
        for key, value in sorted(pv.iteritems()):
            f.write('%s %s\n' % (key, str(value)))


def cp_files(d0):
    cmd = 'cp %s/%s %s/%s'
    os.system(cmd % (dpd_dir, 'test',               d0, ''))
    os.system(cmd % (dpd_dir, 'sdf/wall1/wall.dat', d0, 'sdf.dat'))
    rc = pv['RBCrc']
    if   rc == 1: rf = 'rbc.r1.dat'
    elif rc == 2: rf = 'rbc.r2.dat'
    elif rc == 4: rf = 'rbc.r4.dat'
    elif rc == 8: rf = 'rbc.r8.dat'
    else: print 'Resolution not found!'; sys.exit()
    os.system(cmd % (dpd_dir, rf, d0, 'rbc.dat'))


def gen_par(pn0, pv0):
    set_defaults()
    for j in range(len(pv0)): pv[pn0[j]] = float(pv0[j])

    pv['gammadpd_rbc'] = pv['gammadpd_wall'] = pv['gammadpd_out']

    sh = pv['gamma_dot']
    if sh > 0:
        pv['tend'] = 3*800/sh
        pv['steps_per_dump'] = pv['steps_per_hdf5dump'] = int(3*800/sh)

    if pv['stretchingForce'] > 0:
        pv['nosolvent'] = 1
        pv['RBCrnd'] = 0

    rc = pv['RBCrc']
    if   rc == 1: nv = 498;   nt = 992
    elif rc == 2: nv = 1986;  nt = 3968
    elif rc == 4: nv = 7938;  nt = 15872
    elif rc == 8: nv = 31746; nt = 63488
    pv['RBCnv'] = nv; pv['RBCnt'] = nt

    t = sqrt(3.)*(pv['RBCnv']-2)
    t = acos((t - 5*pi) / (t - 3*pi))
    pv['RBCphi'] = 180./pi * t

    # warning: scaling is opposite to rc
    pv['XS']           *= rc
    pv['YS']           *= rc
    pv['ZS']           *= rc
    pv['RBCp']         *= rc
    pv['RBCtotArea']   *= rc*rc
    pv['RBCtotVolume'] *= rc*rc*rc
    pv['strVerts']     *= rc


def recompile():
    cmd = 'cd %s && make clean && make -j > make.log'
    os.system(cmd % dpd_dir)


def pre():
    d0 = gen_dir()
    write_par_file(d0)
    cp_files(d0)
    gen_ic(d0)
    return d0


def run_falcon(d0):
    os.system('cd %s && ./test' % d0)


def run_daint_i(d0):
    os.system('cd %s && srun ./test' % d0)


def run_daint(d0):
    with open('%s/runme.sh' % d0, 'w') as f:
        f.write('#!/bin/bash -l\n')
        f.write('#SBATCH --job-name=%s\n' % d0)
        f.write('#SBATCH --time=6:00:00\n')
        f.write('#SBATCH --nodes=1\n')
        f.write('#SBATCH --ntasks-per-node=1\n')
        f.write('#SBATCH --output=output.txt\n')
        f.write('#SBATCH --error=error.txt\n')
        f.write('#SBATCH --constraint=gpu\n')
        f.write('#SBATCH --account=s659\n')
        f.write('module load cudatoolkit\n')
        f.write('module load cray-hdf5-parallel\n')
        f.write('export HEX_COMM_FACTOR=2\n')
        f.write('srun --export ALL ./test 1 1 1\n')
        f.write('sh run_post.parallel.sh .\n')
    os.system('cd %s && sbatch runme.sh' % d0)


def run(d0, machine):
    if machine == 'falcon':
        run_falcon(d0)
    elif machine == 'daint':
        run_daint(d0)
    elif machine == 'daint_i':
        run_daint_i(d0)
    else:
        print 'Unknown machine'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--machine', default='falcon')
    args = parser.parse_args()
    machine = args.machine

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
        run(d0, machine)
