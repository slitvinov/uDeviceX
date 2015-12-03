#!/usr/bin/awk -f


function advance() {      # lexical analyzer; returns next token
    if (tok == "(eof)") return "(eof)"    
    if (length(line) == 0)
        if (getline line == 0)
            return tok = "(eof)"
	else
	    return tok = sep
    sep = "\n"

    if (match(line, funcall) ||
	match(line, /^./)) {                    # everything else
	tok = substr(line, 1, RLENGTH)
	line = substr(line, RLENGTH+1)
	return tok
    }
}

BEGIN {
    identifier = "[A-Za-z_][A-Za-z_0-9]*"
    funcall    = "^" identifier "[\t ]*[\(]"
    
    do
	advance()
    while (tok != "(eof)" && tok !~ funcall)

    fn = tok         # function name
    sub("[(][\t ]*$", "", fn)

    for (;;) {
	advance()
	if (tok == "(eof)" || tok == ")") break
	args = args tok
    }
    
    printf "%s(%s):=", fn, args

    for (;;) {
	advance()
	if (tok == "(eof)") break
	rst = rst tok
    }
    
    print rst
}
