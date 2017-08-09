namespace cnt {
void bulk(std::vector<ParticlesWrap> wsolutes) {
    if (wsolutes.size() == 0) return;

    for (int i = 0; i < (int) wsolutes.size(); ++i) {
        ParticlesWrap it = wsolutes[i];
        KL(k_cnt::bulk, (k_cnf(3 * it.n)),
           ((float2 *)it.p, it.n, cellsentries->S, wsolutes.size(), (float *)it.f,
            rgen->get_float(), i));
    }
}
}
