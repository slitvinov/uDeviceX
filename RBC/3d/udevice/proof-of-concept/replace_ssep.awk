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

function error(s) { print "Error: " s | "cat 1>&2"; exit 1 }

function eat(s) {     # read next token if s == tok
    if (tok != s) error("line " NF ": saw " tok ", expected " s)
    advance()
}

function expr(   e) {
    for (;;) {
	if      (tok == ";")     {eat(";"); break}
	else if (tok == "(eof)") break
	else if (tok == "}")     break
	else if (tok == "{")     e = e "(" exprlist() ")\n"        # TODO: why \n?
	else     e = e tok
	advance()
    }
    return e
}

function exprlist(   e, c, sep) {
    eat("{")
    for (;;) {
	c   = expr()
	sep =  e  && c !~ /^[ \t\n]*$/ ? "," : ""
	e   = e sep c
	if (tok == "}")     {eat("}"); return e}
	if (tok == "(eof)")  return e
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
