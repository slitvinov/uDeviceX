namespace rex {
bool post_pre(MPI_Comm cart, int ranks[26], int tags[26]) {
    bool packingfailed;
    int i;

    dSync();
    if (iterationcount == 0) _postrecvC(cart, ranks, tags);
    else _wait(reqsendC);

    for (i = 0; i < 26; ++i) send_counts[i] = host_packstotalcount->D[i];
    packingfailed = false;
    for (i = 0; i < 26; ++i)
        packingfailed |= send_counts[i] > local[i]->capacity();
    return packingfailed;
}

void post_resize() {
    int newcapacities[26];
    int *newindices[26];
    int i;

    for (i = 0; i < 26; ++i) local[i]->resize(send_counts[i]);
    for (i = 0; i < 26; ++i) newcapacities[i] = local[i]->capacity();
    CC(cudaMemcpyToSymbolAsync(k_rex::ccapacities, newcapacities,
                               sizeof(newcapacities), 0,
                               H2D));
    for (i = 0; i < 26; ++i) newindices[i] = local[i]->scattered_indices->D;
    CC(cudaMemcpyToSymbolAsync(k_rex::scattered_indices, newindices,
                               sizeof(newindices), 0, H2D));
}

void post_p(MPI_Comm cart, int ranks[26], int tags[26]) {
    // consolidate the packing
    {
        for (int i = 0; i < 26; ++i) local[i]->resize(send_counts[i]);

        _postrecvA(cart, ranks, tags);

        if (iterationcount == 0) {
            _postrecvP(cart, ranks, tags);
        } else
            _wait(reqsendP);

        if (host_packstotalstart->D[26]) {
            CC(cudaMemcpyAsync(host_packbuf->D, packbuf->D,
                               sizeof(Particle) * host_packstotalstart->D[26],
                               H2H));
        }
        dSync(); /* was CC(cudaStreamSynchronize(downloadstream)); */
    }

    // post the sending of the packs
    {
        reqsendC.resize(26);

        for (int i = 0; i < 26; ++i)
            MC(l::m::Isend(send_counts + i, 1, MPI_INTEGER, ranks[i],
                           btc + i, cart, &reqsendC[i]));

        for (int i = 0; i < 26; ++i) {
            int start = host_packstotalstart->D[i];
            int count = send_counts[i];
            int expected = local[i]->expected();

            MPI_Request reqP;
            MC(l::m::Isend(host_packbuf->D + start, expected * 6, MPI_FLOAT,
                           ranks[i], btp1 + i, cart, &reqP));
            reqsendP.push_back(reqP);

            if (count > expected) {
                MPI_Request reqP2;
                MC(l::m::Isend(host_packbuf->D + start + expected,
                               (count - expected) * 6, MPI_FLOAT, ranks[i],
                               btp2 + i, cart, &reqP2));

                reqsendP.push_back(reqP2);
            }
        }
    }
}
}
