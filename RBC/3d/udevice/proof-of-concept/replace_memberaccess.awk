#!/usr/bin/awk -f

function advance() {      # lexical analyzer; returns next token
    if (tok == "(eof)") return "(eof)"    
    if (length(line) == 0)
        if (getline line == 0)
            return tok = "(eof)"
	else
	    return tok = sep
    sep = "\n"

    if (match(line, member_of_object) ||
	match(line, /^./)) {                    # everything else
	tok = substr(line, 1, RLENGTH)
	line = substr(line, RLENGTH+1)
	return tok
    }
}

BEGIN {
    identifier = "[A-Za-z_][A-Za-z_0-9]*"
    member_of_object = "^" identifier "[.]" identifier
    
    for (;;) {
	advance()
	if (tok == "(eof)") break	

	sym = tok
	if (tok ~ member_of_object)
	    sub("[.]", "%", sym)
	ans = ans sym
    }
    print ans
}
