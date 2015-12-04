#!/usr/bin/awk -f

function advance() {      # lexical analyzer; returns next token
    if (tok == "(eof)") return "(eof)"    
    if (length(line) == 0)
        if (getline line == 0)
            return tok = "(eof)"
	else
	    return tok = sep
    sep = "\n"

    if (match(line, "^" identifier) ||
	match(line, float)          ||
	match(line, /^./)) {                    # everything else
	tok = substr(line, 1, RLENGTH)
	line = substr(line, RLENGTH+1)
	return tok
    }
}

function float2rat(e,   ans) {
    cmd = "proof-of-concept/rat.sh " e
    cmd | getline ans

    print "(rat_float.awk) " e "->" ans > "/dev/stderr"

    close(cmd)
    return ans
}

BEGIN {
    float="^[-]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][+-]?[0-9]+)?"
    identifier = "[A-Za-z_][A-Za-z_0-9]*"
    
    for (;;) {
	advance()
	if (tok == "(eof)") break	

	sym = tok
	if (tok ~ float "$") {
	    sym = float2rat(sym)
	}

	ans = ans sym
    }
    print ans
}

