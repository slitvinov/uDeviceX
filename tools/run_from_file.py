#!/usr/bin/env python
import os
import sys


def parse_file(fname, pv, pos):
    with open(fname, 'r') as f: lines = f.readlines()
    with open(fname, 'w') as f:
        for l in lines:
            w = l.split()
            if len(w) > 2 and w[1] in pv:
                for k in range(0, pos): f.write('%s ' % w[k])
                f.write('%g ' % pv[w[1]])
                for k in range(pos+1, len(w)): f.write('%s ' % w[k])
                f.write('\n')
            else: f.write(l)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: %s <start line> <end line>' % sys.argv[0]
        exit()

    home = os.path.expanduser('~')
    dpd_dir = home+'/rbc_shear_tags/mpi-dpd'
    cnf_file = dpd_dir+'/.conf.test.h'
    rbc_file = dpd_dir+'/params/rbc.inc0.h'
    tools = home+'/rbc_shear_tags/tools'

    with open('points.txt', 'r') as f: lines = f.readlines()
    pn = ['rc', '_numberdensity', '_aij_out', '_aij_in', '_aij_rbc', '_aij_wall',
            '_gammadpd_out', '_gammadpd_in', '_gammadpd_rbc', '_gammadpd_wall',
            '_kBT', '_gamma_dot', 'XS', 'YS', 'ZS', 'dt',
            'RBCgammaC', 'RBCkb', 'RBCp', 'RBCx0', 'RBCkd']

    d0 = 'simulations'
    if not os.path.exists(d0): os.makedirs(d0)

    st = int(sys.argv[1])-1; fi = int(sys.argv[2])-1  # 0-base numbering
    for i in range(st, fi+1):
        print 'Running case', i+1

        pv = {}; pv0 = lines[i].split()
        for j in range(len(pv0)): pv[pn[j]] = float(pv0[j])

        stpd = int(1600/pv['_gamma_dot'])
        tend = int(80/pv['_gamma_dot'])

        # replace corresponsing entries in config file
        parse_file(cnf_file, pv, 2)  # last arg -- 0-base index of param value
        parse_file(rbc_file, pv, 3)  # last arg -- 0-base index of param value

        cmd = ('export PATH=%s:$PATH && argp %s -rbcs -tend=%f' +
                ' -walls -wall_creation_stepid=100' +
                ' -hdf5field_dumps -hdf5part_dumps' +
                ' -steps_per_dump=%d -steps_per_hdf5dump=%d > %s/.conf.h')
        os.system(cmd % (tools, cnf_file, tend, stpd, stpd, dpd_dir))

        # recompile
        cmd = 'cd %s && make clean && make -j'
        os.system(cmd % dpd_dir)

        # create working directory
        d = d0+'/'
        for j in range(len(pv)): d += ('%s_%g_' % (pn[j], pv[pn[j]]))
        d = d[0:-2]  # remove the last '_'
        if not os.path.exists(d): os.makedirs(d)

        # write params file
        with open(d+'/params.txt', 'w') as f:
            for t in pn: f.write(str(pv[t])+' ')

        # copy necessary files
        cmd = 'cp %s/%s %s/%s'
        os.system(cmd % (dpd_dir, 'test', d, ''))
        os.system(cmd % (dpd_dir, 'sdf/wall1/wall.dat', d, 'sdf.dat'))
        os.system(cmd % (dpd_dir, 'rbc.dat', d, ''))

        # IC for RBC
        with open(d+'/rbcs-ic.txt', 'w') as f:
            f.write('1 0 0 %g  0 1 0 %g  0 0 1 %g  0 0 0 1\n' % (pv['XS']/2, pv['YS']/2, pv['ZS']/2))

        # run
        os.system('cd %s && ./test' % d)
        os.system('cd %s/ply && sh ~/scripts/ply/cm.sh' % d)

        # postprocessing
        sys.path.append(dpd_dir)
        import compute_freq_standalone_Tran_Son_Tay as cT
        import numpy as np

        time, theta, omega = cT.read_data(d+'/ply', pv['dt'], stpd)
        try:
            fr, fru = cT.get_fr(time, omega)
            an, anu = cT.get_an(theta)

            with open(d+'/post.txt', 'w') as f:
                sh = pv['_gamma_dot']
                f.write('freq/shrate: %g %g\n' % (2.*np.pi*fr/sh, fru/sh))
                f.write('angle: %g %g\n' % (an, anu))
        except:
            pass
