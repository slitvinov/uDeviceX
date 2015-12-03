#!/usr/bin/awk -f

BEGIN {
    float="^[+-]?([0-9]+[.]?[0-9]*|[.][0-9]+)([eE][+-]?[0-9]+)?"
    ff   = float "f"

    identifier = "[A-Za-z_][A-Za-z_0-9]*"
}

function advance() {      # lexical analyzer; returns next token
    if (tok == "(eof)") return "(eof)"    
    if (length(line) == 0)
        if (getline line == 0)
            return tok = "(eof)"
	else
	    return tok = sep
    sep = "\n"

    if (match(line, "^" identifier) ||
	match(line, ff)             ||
	match(line, /^./)) {                    # everything else
	tok = substr(line, 1, RLENGTH)
	line = substr(line, RLENGTH+1)
	return tok
    }
}

BEGIN {
    for (;;) {
	advance()
	if (tok == "(eof)") break	

	sym = tok
	if (tok ~ ff)
	    sym = substr(tok, 1, length(tok)-1) # cut `f'

	ans = ans sym
    }
    print ans
}

