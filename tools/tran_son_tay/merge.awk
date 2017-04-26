#!/usr/bin/awk -f

/^#/ {next}
{
    parse()
    print a*cos(d2r(th)), acth_star, s_acth_star, c, c_star, s_c_star
}

function parse(   i) {
    i=1
    gd = $(i++)
    acth_star = $(i++)
    s_acth_star = $(i++)
    c_star = $(i++)
    s_c_star = $(i++)
    f = $(i++)
    s_f = $(i++)
    gd = $(i++)
    a = $(i++)
    b = $(i++)
    c = $(i++)
    th = $(i++)
}

function d2r(x) { return x*pi/180 }
