void fin_rnd(rbc::rnd::D *rnd) {
    rbc::rnd::fin(rnd);
}

void fin_ticket(TicketT *t) {
    destroy(&t->textri);
    destroy(&t->texadj0);
    destroy(&t->texadj1);
    destroy(&t->texvert);
    if (RBC_RND) fin_rnd(t->rnd);
}