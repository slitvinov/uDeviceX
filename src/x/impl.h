namespace x {
void ini(/*io*/ basetags::TagGen *tg) {
    cnt = -1; /* TODO: */
    ini_ticketcom(&tc);
    ini_ticketr(&tr);
    ini_tickettags(tg, &tt);
    rex::ini();
}

void fin() {
    rex::fin();
    fin_ticketcom(tc);
}

static void post(TicketCom tc, TicketR tr, x::TicketTags t, std::vector<ParticlesWrap> w) {
    bool packingfailed;
    dSync();
    if (cnt == 0) rex::_postrecvC(tc.cart, tc.ranks, tr.tags, t);
    else          rex::post_waitC();
    packingfailed = rex::post_pre(tc.cart, tc.ranks, tr.tags, t);
    if (packingfailed) {
        rex::post_resize();
        rex::_adjust_packbuffers();
        rex::_pack_attempt(w);
        dSync();
    }
    rex::local_resize();
    rex::_postrecvA(tc.cart, tc.ranks, tr.tags, t);

    if (cnt == 0) rex::_postrecvP(tc.cart, tc.ranks, tr.tags, t);
    else          rex::post_waitP();
    rex::post_p(tc.cart, tc.ranks, tr.tags, t);
}

static void rex0(std::vector<ParticlesWrap> w, int nw) {
    cnt++;
    rex::pack_p(nw);
    rex::_pack_attempt(w);
    post(tc, tr, tt, w);
    rex::recv_p(tc.cart, tc.ranks, tr.tags, tt);
    if (cnt) rex::halo_wait();
    rex::halo(); /* fsi::halo(); */
    rex::_postrecvP(tc.cart, tc.ranks, tr.tags, tt);
    rex::post_f(tc.cart, tc.ranks, tt);
    rex::recv_f(w);
}

void rex(std::vector<ParticlesWrap> w) {
    int nw;
    nw = w.size();
    if (nw) rex0(w, nw);
}

}
