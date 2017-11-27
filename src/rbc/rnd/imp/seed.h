static long decode_seed_env() {
    char *s;
    s = getenv("RBC_RND");
    if   (s == NULL) {
        MSG("RBC_RND is not set");
        return 0;
    }
    else  {
        MSG("RBC_RND = %s", s);
        return atol(s);
    }
}
static long decode_seed_time() {
    long t;
    t = os::time();
    MSG("t: %ld", t);
    return t;
}
static int decode_seed(long seed) {
    if      (seed == ENV )  return decode_seed_env();
    else if (seed == TIME) return decode_seed_time();
    else return seed;
}