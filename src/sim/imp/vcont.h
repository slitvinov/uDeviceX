/* velocity controller */

static bool valid_step(long id, const int freq) {
    return (freq != 0) && (id % freq == 0);
}

void sample(const Coords *coords, long id, const Flu *f, /**/ Vcon *c) {
    if (valid_step(id, c->sample_freq)) {
        const FluQuants *q = &f->q;
        vcont_sample(coords, q->n, q->pp, q->cells.starts, q->cells.counts, /**/ c->vcont);
    }
}

void adjust(long id, /**/ Vcon *c, BForce *bforce) {
    if (valid_step(id, c->adjust_freq)) {
        float3 f;
        f = vcont_adjustF(/**/ c->vcont);
        bforce_adjust(f, /**/ bforce);
    }
}

void log(long id, const Vcon *c) {
    if (valid_step(id, c->log_freq))
        vcont_log(c->vcont);
}
