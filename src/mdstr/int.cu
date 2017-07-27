#include <mpi.h>
#include "m.h"
#include "l/m.h"
#include "basetags.h"

#include "common.h"

#include "mdstr/imp.h"
#include "mdstr/int.h"

namespace mdstr {

void ini_ticketC(/*io*/ basetags::TagGen *tg, /**/ TicketC *t) {
    l::m::Comm_dup(m::cart, &t->cart);
    sub::gen_ne(m::cart, t->rnk_ne, t->ank_ne);
    t->first = true;
    t->btc = get_tag(tg);
    t->btp = get_tag(tg);
}

void free_ticketC(/**/ TicketC *t) {
    l::m::Comm_free(&t->cart);
}

void ini_ticketS(int nv, /**/ TicketS *t) {
    for (int i = 0; i < 27; ++i) t->pp[i] = new Particle[MAX_PART_NUM];
    for (int i = 0; i < 27; ++i) t->counts[i] = 0;
}

void free_ticketS(/**/ TicketS *t) {
    for (int i = 0; i < 27; ++i) delete[] t->pp[i];
}

void ini_ticketR(int nv, const TicketS *ts, /**/ TicketR *t) {
    t->pp[0] = ts->pp[0]; // bulk
    for (int i = 1; i < 27; ++i) t->pp[i] = new Particle[MAX_PART_NUM];
    for (int i = 0; i < 27; ++i) t->counts[i] = 0;
}

void free_ticketR(/**/ TicketR *t) {
    for (int i = 1; i < 27; ++i) delete[] t->pp[i];
}

void pack();
void post();
void wait();
void unpack();

#undef i2del
} // mdstr
