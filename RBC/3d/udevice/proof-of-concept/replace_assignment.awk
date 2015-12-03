#!/usr/bin/awk -f

function advance() {      # lexical analyzer; returns next token
    if (tok == "(eof)") return "(eof)"    
    if (length(line) == 0)
        if (getline line == 0)
            return tok = "(eof)"
	else
	    return tok = sep
    sep = "\n"

        # http://en.cppreference.com/w/cpp/language/operator_assignment
    if 	(match(line, /^[+]=/)  ||
	match(line, /^[-]=/)  ||
	match(line, /^[\*]=/) ||	
	match(line, /^[\/]=/) ||
	match(line, /^[%]=/)  ||
	match(line, /^[&]=/)  ||
	match(line, /^[|]=/)  ||
	match(line, /^<<=/)   ||
	match(line, /^>>=/)   ||

	# http://en.cppreference.com/w/cpp/language/operator_comparison
	match(line, /^==/)    ||
        match(line, /^!=/)    ||
	match(line, /<=/)     ||
	match(line, />=/)     ||

	# http://en.cppreference.com/w/cpp/language/operator_logical
	match(line,  /^\&\&/)  ||
	match(line,  /^\|\|/)  ||
	match(line, /^!/)      ||	 
       
	match(line, /^./)) {                    # everything else
	tok = substr(line, 1, RLENGTH)
	line = substr(line, RLENGTH+1)
	return tok
    }
}


BEGIN {
    op["=" ]  = ":"
    op["==" ] = "="    
    op["!="]  = "#"
    
    op["&&"]  = " and "
    op["||"]  = " or "
    op["!"]   = "not "    
    
    for (;;) {
	advance()
	if (tok == "(eof)") break
	
	if (tok in op)
	    ans = ans op[tok]
	else
	    ans = ans tok
    }
    print ans
}
