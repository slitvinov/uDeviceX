#!/usr/bin/awk -f

# Remove "//" comments

function asplit(str, arr,    temp) {  # make an assoc array from str
    n = split(str, temp)
    for (i = 1; i <= n; i++)
        arr[temp[i]]++
    return n
}

function advance() {      # lexical analyzer; returns next token
    if (tok == "(eof)") return "(eof)"    
    if (length(line) == 0)
        if (getline line == 0)
            return tok = "(eof)"
	else
	    return tok = sep
    sep = "\n"

    if (match(line, /^\/\//) ||
	match(line, /^./)) {                    # everything else
	tok = substr(line, 1, RLENGTH)
	line = substr(line, RLENGTH+1)
	return tok
    }
}

function eat_comments() {
    while (tok != "(eof)" && tok != "\n" )
	advance()
}

BEGIN {
    for (;;) {
	advance()
	if (tok == "//")
	    eat_comments()
	if (tok == "(eof)") break
	ans = ans tok
	
    }
    print ans
}
