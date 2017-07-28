namespace rdstr {

using mdstr::TicketC;

/* ticket extend */
struct TicketE {
    PinnedHostBuffer2<float3> *ll, *hh;    /* extents of RBCs                   */
    float *rr;                             /* positions used to distribute rbcs */
};

using mdstr::ini_ticketC;
using mdstr::free_ticketC;

// TODO change them to custom one
using mdstr::TicketS;
using mdstr::TicketR;

void alloc_ticketE(/**/ TicketE *t);
void free_ticketE(/**/ TicketE *t);

// TODO custom version
using mdstr::ini_ticketS;
using mdstr::ini_ticketR;
using mdstr::free_ticketS;
using mdstr::free_ticketR;


void extents(const Particle *pp, int nc, int nv, /**/ TicketE *t);
void get_pos(int nc, /**/ TicketE *t);

using mdstr::get_reord;

// TODO: custom ones
using mdstr::pack;
using mdstr::post_send;
using mdstr::post_recv;
using mdstr::wait_recv;
using mdstr::unpack;

} // rdstr
