namespace sim {
static std::vector<Particle> ic_pos() { /* generate particle position */
  srand48(0);
  std::vector<Particle> pp;
  int L[3] = {XS, YS, ZS};
  int iz, iy, ix, l, nd = numberdensity;
  float x, y, z, dr = 0.99;
  for (iz = 0; iz < L[2]; iz++)
    for (iy = 0; iy < L[1]; iy++)
      for (ix = 0; ix < L[0]; ix++) {
	/* edge of a cell */
	int xlo = -L[0]/2 + ix, ylo = -L[1]/2 + iy, zlo = -L[2]/2 + iz;
	for (l = 0; l < nd; l++) {
	  Particle p = Particle();
	  x = xlo + dr * drand48(), y = ylo + dr * drand48(), z = zlo + dr * drand48();
	  p.r[0] = x; p.r[1] = y; p.r[2] = z;
	  p.v[0] = 0; p.v[1] = 0; p.v[2] = 0;
	  pp.push_back(p);
	}
      }
  fprintf(stderr, "(simulation) generated %d particles\n", pp.size());
  return pp;
}

static void redistribute() {
  sdstr::pack(s_pp, s_n);
  if (rbcs) rdstr::extent(r_pp, r_nc, r_nv);
  sdstr::send();
  if (rbcs) rdstr::pack_sendcnt(r_pp, r_nc, r_nv);
  sdstr::bulk(s_n, cells->start, cells->count);
  s_n = sdstr::recv_count();
  if (rbcs) {
    r_nc = rdstr::post(r_nv); r_n = r_nc * r_nv;
  }
  sdstr::recv_unpack(s_pp0, s_zip0, s_zip1, s_n, cells->start, cells->count);
  std::swap(s_pp, s_pp0); std::swap(s_ff, s_ff0);
  if (rbcs) rdstr::unpack(r_pp, r_nc, r_nv);
}

void remove_bodies_from_wall() {
  if (!rbcs) return;
  if (!r_nc) return;
  DeviceBuffer<int> marks(r_n);
  k_wall::fill_keys<<<(r_n + 127) / 128, 128>>>
    (r_pp, r_n, marks.D);

  std::vector<int> tmp(marks.S);
  CC(cudaMemcpy(tmp.data(), marks.D, sizeof(int) * marks.S, D2H));
  std::vector<int> tokill;
  for (int i = 0; i < r_nc; ++i) {
    bool valid = true;
    for (int j = 0; j < r_nv && valid; ++j)
      valid &= 0 == tmp[j + r_nv * i];
    if (!valid) tokill.push_back(i);
  }

  r_nc = Cont::rbc_remove(r_pp, r_nv, r_nc, &tokill.front(), tokill.size());
  r_n = r_nc * r_nv;
}

static void update_helper_arrays() {
  CC(cudaFuncSetCacheConfig(k_sim::make_texture, cudaFuncCachePreferShared));
  k_sim::make_texture<<<(s_n + 1023) / 1024, 1024, 1024 * 6 * sizeof(float)>>>
    (s_zip0, s_zip1, (float*)s_pp, s_n);
}

void create_walls() {
  dSync();
  s_n = wall::init(s_pp, s_n); /* number of survived particles */
  wall_created = true;

  Cont::ic_shear_velocity(s_pp, s_n);
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();

  // remove cells touching the wall
  remove_bodies_from_wall();
}

void forces_rbc() {
  if (rbcs) rbc::forces_nohost(r_nc, (float*)r_pp, (float*)r_ff,
			       r_host_av, r_addfrc);
}

void forces_dpd() {
  DPD::pack(s_pp, s_n, cells->start, cells->count);
  DPD::local_interactions(s_pp, s_zip0, s_zip1,
			  s_n, s_ff, cells->start,
			  cells->count);
  DPD::post(s_pp, s_n);
  DPD::recv();
  DPD::remote_interactions(s_n, s_ff);
}

void clear_forces() {
  Cont::clear_forces(s_ff, s_n);
  if (rbcs) Cont::clear_forces(r_ff, r_n);
}

void forces_wall() {
  if (rbcs && wall_created) wall::interactions(r_pp, r_n, r_ff);
  if (wall_created)         wall::interactions(s_pp, s_n, s_ff);
}

void forces_cnt(std::vector<ParticlesWrap> *w_r) {
  if (contactforces) {
    cnt::build_cells(*w_r);
    cnt::bulk(*w_r);
  }
}

void forces_fsi(SolventWrap *w_s, std::vector<ParticlesWrap> *w_r) {
  fsi::bind_solvent(*w_s);
  fsi::bulk(*w_r);
}

void forces() {
  SolventWrap w_s(s_pp, s_n, s_ff, cells->start, cells->count);
  std::vector<ParticlesWrap> w_r;
  if (rbcs) w_r.push_back(ParticlesWrap(r_pp, r_n, r_ff));

  clear_forces();

  forces_dpd();
  forces_wall();
  forces_rbc();

  forces_cnt(&w_r);
  forces_fsi(&w_s, &w_r);

  rex::bind_solutes(w_r);
  rex::pack_p();
  rex::post_p();
  rex::recv_p();

  rex::halo(); /* fsi::halo(); cnt::halo() */

  rex::post_f();
  rex::recv_f();
}

void rm_vmean() {
   Cont::rm_vmean(s_pp, s_n, r_pp, r_n);
}

void in_out() {
#ifdef GWRP
#include "sim.hack.h"
#endif
}

void dev2hst() { /* device to host  data transfer */
  CC(cudaMemcpyAsync(sr_pp, s_pp,
		     sizeof(Particle) * s_n, D2H, 0));
  if (rbcs)
    CC(cudaMemcpyAsync(&sr_pp[s_n], r_pp,
		       sizeof(Particle) * r_n, D2H, 0));
}

bool in_box(float *r) {
  enum {X, Y, Z};
  float xlo = -12, xhi = -xlo,
        ylo = -5,  yhi = -ylo,
        zlo = -5,  zhi = -zlo;
  return (   xlo <= r[X] && r[X] <= xhi
          && ylo <= r[Y] && r[Y] <= yhi
          && zlo <= r[Z] && r[Z] <= zhi);
}

int mv_box(Particle *pp, int n) {
  Particle p;
  int i, i0 = 0 /* first particle not in box */;
  for (i = 0; i < n; ++i) {
    float *r = pp[i].r;
    if (in_box(r)) {
      p = pp[i];
      pp[i] = pp[i0];
      pp[i0] = p;
      i0++;
    }
  }
  return i0;
}

void dump_part() {
  if (!hdf5part_dumps) return;
  dev2hst(); /* TODO: do not need `r' */
  int n = mv_box(sr_pp, s_n);
  dump_part_solvent->dump(sr_pp, n);
}

void dump_rbcs() {
  if (!rbcs) return;
  static int id = 0;
  dev2hst();  /* TODO: do not need `s' */
  Cont::rbc_dump(r_nc, &sr_pp[s_n], triplets, r_n, r_nv, r_nt, id++);
}

void dump_grid() {
  if (!hdf5field_dumps) return;
  dev2hst();  /* TODO: do not need `r' */
  dump_field->dump(sr_pp, s_n);
}

void diag(int it) {
  int n = s_n + r_n; dev2hst();
  diagnostics(sr_pp, n, it);
}

void update() {
  Cont::update(s_pp, s_ff, s_n, false, driving_force);
  if (rbcs) Cont::update(r_pp, r_ff, r_n, true, driving_force);
}

void bounce() {
  if (!wall_created) return;
  wall::bounce(s_pp, s_n);
  if (rbcs) wall::bounce(r_pp, r_n);
}

void init() {
  CC(cudaMalloc(&r_orig_xyzuvw, RBCnv * 6 * sizeof(float)));
  CC(cudaMalloc(&r_host_av, MAX_CELLS_NUM));
  CC(cudaMalloc(&r_addfrc, RBCnv * sizeof(float)));

  
  rbc::setup(triplets, r_orig_xyzuvw, r_addfrc);
  rdstr::init();
  DPD::init();
  fsi::init();
  rex::init();
  cnt::init();
  if (hdf5part_dumps)
    dump_part_solvent = new H5PartDump("s.h5part");

  cells   = new CellLists(XS, YS, ZS);
  mpDeviceMalloc(&s_zip0); mpDeviceMalloc(&s_zip1);

  if (rbcs) {
      mpDeviceMalloc(&r_pp); mpDeviceMalloc(&r_ff);
  }

  wall::trunk = new Logistic::KISS;
  sdstr::init();
  mpDeviceMalloc(&s_pp); mpDeviceMalloc(&s_pp0);
  mpDeviceMalloc(&s_ff); mpDeviceMalloc(&s_ff0);
  mpDeviceMalloc(&r_ff); mpDeviceMalloc(&r_ff);


  std::vector<Particle> ic = ic_pos();
  s_n  = ic.size();

  CC(cudaMemcpy(s_pp, &ic.front(), sizeof(Particle) * ic.size(), H2D));
  cells->build(s_pp, s_n, NULL, NULL);
  update_helper_arrays();

  if (rbcs) {
    r_nc = Cont::setup(r_pp, r_nv, "rbcs-ic.txt", r_orig_xyzuvw);
    r_n = r_nc * r_nv;
#ifdef GWRP
    iotags_init_file("rbc.dat");
    iotags_domain(0, 0, 0,
		  XS, YS, ZS,
		  m::periods[0], m::periods[1], m::periods[0]);
#endif
  }

  dump_field = new H5FieldDump;
  MC(MPI_Barrier(m::cart));
}

void dumps_diags(int it) {
  if (it % steps_per_dump == 0)     in_out();
  if (it % steps_per_dump == 0)     dump_rbcs();
  if (it % steps_per_dump == 0)     dump_part();
  if (it % steps_per_hdf5dump == 0) dump_grid();
  if (it % steps_per_dump == 0)     diag(it);
}

void run() {
  int nsteps = (int)(tend / dt);
  if (m::rank == 0 && !walls) printf("will take %ld steps\n", nsteps);
  if (!walls && pushtheflow) driving_force = hydrostatic_a;
  int it;
  for (it = 0; it < nsteps; ++it) {
    if (walls && it == wall_creation_stepid) {
      create_walls();
      if (rbcs) Cont::ic_shear_velocity(r_pp, r_n);
      if (pushtheflow) driving_force = hydrostatic_a;
    }
    redistribute();
    rm_vmean();
    forces();
    dumps_diags(it);
    update();
    bounce();
  }
}

void close() {
  delete dump_field;
  delete dump_part_solvent;
  sdstr::redist_part_close();

  cnt::close();
  delete cells;
  rex::close();
  fsi::close();
  DPD::close();
  rdstr::close();

  CC(cudaFree(s_zip0));
  CC(cudaFree(s_zip1));

  CC(cudaFree(r_orig_xyzuvw));
  CC(cudaFree(r_host_av));

  delete wall::trunk;
  CC(cudaFree(r_pp )); CC(cudaFree(r_ff ));
  CC(cudaFree(s_pp )); CC(cudaFree(s_ff ));
  CC(cudaFree(s_pp0)); CC(cudaFree(s_ff0));
}
}
