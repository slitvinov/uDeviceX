#include <stdio.h>
#include "utils/imp.h"
#include "utils/error.h"
#include "utils/msg.h"
#include "frag/imp.h"

#include "imp.h"

enum {SUCCESS, PACK_FAILURE};
struct DFluStatus {
    int errorcode;
    int cap, cnt; /* capacity and count */
    int fid;      /* fragment id */
};

void dflu_status_ini(DFluStatus **ps) {
    DFluStatus *s;
    UC(emalloc(sizeof(DFluStatus), (void**)&s));
    s->errorcode = SUCCESS;
    *ps = s;
}

void dflu_status_fin(DFluStatus *s) {
    UC(efree(s));
}

int  dflu_status_success(DFluStatus *s) {
    return s->errorcode == SUCCESS;
}

static void success() { msg_print("DFluStatus: SUCCESS"); }
static void pack_failure(DFluStatus *s) {
    enum {X, Y, Z};
    int cap, cnt, fid, d[3];
    cap = s->cap; cnt = s->cnt; fid = s->fid;
    d[X] = frag_i2dx(fid); d[Y] = frag_i2dy(fid); d[Z] = frag_i2dz(fid);
    msg_print("exceed capacity, fragment %d = [%d %d %d]: %ld/%ld",
        fid, d[X], d[Y], d[Z], cnt, cap);
}
void dflu_status_log(DFluStatus *s) {
    int code;
    if (s == NULL) success();
    else {
        code = s->errorcode;
        if      (code == SUCCESS)      success();
        else if (code == PACK_FAILURE) pack_failure(s);
        else ERR("unknown errorcode = %d\n", code);
    }
}

void dflu_status_over(int fid, int cnt, int cap, /**/ DFluStatus *s) {
    if (s == NULL) ERR("status == NULL");
    s->fid = fid; s->cnt = cnt; s->cap = cap;
    s->errorcode = PACK_FAILURE;
}
