namespace rex {
struct LocalHalo {
    History h;
    DeviceBuffer<int>* indexes;
    PinnedHostBuffer<Force>* ff;
};

namespace lo {
void resize(LocalHalo *l, int n) {
    l->indexes->resize(n);
    l->ff->resize(n);
}

void update(LocalHalo *l) {
    l->h.update(l->ff->S);
}

int expected(LocalHalo *l) {
    return (int)ceil(l->h.max() * 1.1);
}

int size(LocalHalo *l) {
    return l->indexes->C;
}
}
}
