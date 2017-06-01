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
    pt['RBCrc']                = '#define %s ( %g )\n'
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
    pv['YS']                   = 8
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
    pv['RBCrc']                = 1
    pv['RBCx0']                = 0.45
    pv['RBCp']                 = 0.0039
    pv['RBCka']                = 4900
    pv['RBCkb']                = 32
    pv['RBCkd']                = 200
    pv['RBCkv']                = 5000
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
    pv['hdf5field_dumps']      = 'false'
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
    for d in dd: m = max(m, 1+int(d.split('_')[1]))
    print 'Creating directory #', m
    d0 = '%s/run_%d' % (res_dir, m)
    if not os.path.exists(d0): os.makedirs(d0)
    return d0


def write_par_file(d0):
    with open(d0+'/'+par_file, 'w') as f:
        for key, value in sorted(pv.iteritems()): f.write('%s %s\n' % (key, str(value)))


def cp_files(d0):
    cmd = 'cp %s/%s %s/%s'
    os.system(cmd % (dpd_dir, 'test',               d0, ''))
    os.system(cmd % (dpd_dir, 'sdf/wall1/wall.dat', d0, 'sdf.dat'))
    if pv['RBCrc'] > 1:
        os.system(cmd % (dpd_dir, 'rbc.r1.dat',         d0, 'rbc.dat'))
    else:
        os.system(cmd % (dpd_dir, 'rbc.dat',            d0, 'rbc.dat'))


def gen_par(pn0, pv0):
    set_defaults()
    for j in range(len(pv0)): pv[pn0[j]] = float(pv0[j])

    pv['_gammadpd_rbc'] = pv['_gammadpd_wall'] = pv['_gammadpd_out']

    sh = pv['_gamma_dot']
    pv['tend'] = 800/sh
    pv['steps_per_dump'] = pv['steps_per_hdf5dump'] = int(800/sh)

    if pv['RBCrc'] > 1:
        pv['RBCnv'] = 1986
        pv['RBCnt'] = 3968
    else:
        pv['RBCnv'] = 498
        pv['RBCnt'] = 992

    t = sqrt(3.)*(pv['RBCnv']-2)
    t = acos((t - 5*pi) / (t - 3*pi))
    pv['RBCphi'] = 180./pi * t

    # warning: scaling is opposite to rc
    # pv['RBCkv']        *= RBCrc
    # pv['RBCkb']        *= RBCrc*RBCrc
    # pv['RBCkbT']       *= RBCrc*RBCrc
    pv['XS']           *= RBCrc
    pv['YS']           *= RBCrc
    pv['ZS']           *= RBCrc
    pv['RBCp']         *= RBCrc
    pv['RBCtotArea']   *= RBCrc*RBCrc
    pv['RBCtotVolume'] *= RBCrc*RBCrc*RBCrc


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


def run_daint(d0):
    with open('%s/runme.sh' % d0, 'w') as f:
        f.write('#!/bin/bash -l\n')
        f.write('#SBATCH --job-name=rbc_%s\n' % d0)
        f.write('#SBATCH --time=24:00:00\n')
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
    os.system('cd %s && sbatch runme.sh' % d0)


def run(d0, machine):
    if machine == 'falcon':
        run_falcon(d0)
    elif machine == 'daint':
        run_daint(d0)
    else:
        print 'Unknown machine'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--machine', default='falcon')
    parser.add_argument('--RBCrc', default=1)
    args = parser.parse_args()
    machine = args.machine
    RBCrc = float(args.RBCrc)

    if not os.path.exists(res_dir): os.makedirs(res_dir)

    gen_templates()

    with open(src_file, 'r') as f: lines = f.readlines()
    pn0 = lines[0].replace('#', '').split()  # user-defined parameters names
    pn0.append('RBCrc')

    for l in lines:
        pv0 = l.split()  # user-defined parameters values
        if '#' in pv0[0]: continue  # skip comments

        print 'Running line %s\n' % l

        pv0.append(RBCrc)
        gen_par(pn0, pv0)
        gen_cnf()
        gen_rbc()
        recompile()

        d0 = pre()
        run(d0, machine)
