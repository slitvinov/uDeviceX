#!/usr/bin/awk -f

function advance() {      # lexical analyzer; returns next token
    if (tok == "(eof)") return "(eof)"    
    if (length(line) == 0)
        if (getline line == 0)
            return tok = "(eof)"
	else
	    return tok = sep
    sep = "\n"

    if (match(line, /^{/) ||
	match(line, /^}/) ||
	match(line, /^;/) ||
	match(line, /^./)) {                    # everything else
	tok = substr(line, 1, RLENGTH)
	line = substr(line, RLENGTH+1)
	return tok
    }
}

function ws(c) { # is whitespace?
    return c == " " || c == "\n" || c == "\t"
}

function expr(   e) {
    for (;;) {
	advance()
	if      (tok == "(eof)") break
	else if (tok == ";")     break
	else if (tok == "}")     break	
	else if (tok == "{")     e = e "(" exprlist() ")"
	else     e = e tok
    }
    return e
}

function exprlist(   e, c, sep) {
    e = expr() # at least one expression
    for (  ; tok == ";"  ;) {
	c = expr()
	sep = tok == "}" ? "" : ","
	e = e sep c
    }
    return e
}

BEGIN {
    for (;;) {
	advance()
	if (tok == "(eof)") break

	if (tok == "{")
	    ans = ans "(" exprlist() ")"
	else if (tok != "}")
	    ans = ans tok
    }
    
    print ans "$"
}
