void eflu_post_recv(EFluComm *c, EFluUnpack *u) {
    UC(post_recv(&u->hpp, c->pp));
    UC(post_recv(&u->hfss, c->fss));
    if (multi_solvent)
        UC(post_recv(&u->hcc, c->cc));
}

void eflu_post_send(EFluPack *p, EFluComm *c) {
    UC(post_send(&p->hpp, c->pp));
    UC(post_send(&p->hfss, c->fss));
    if (multi_solvent)
        UC(post_send(&p->hcc, c->cc));
}

void eflu_wait_recv(EFluComm *c, EFluUnpack *u) {
    UC(wait_recv(c->pp, &u->hpp));
    UC(wait_recv(c->fss, &u->hfss));
    if (multi_solvent)
        UC(wait_recv(c->cc, &u->hcc));
}

void eflu_wait_send(EFluComm *c) {
    UC(wait_send(c->pp));
    UC(wait_send(c->fss));
    if (multi_solvent)
        UC(wait_send(c->cc));
}

