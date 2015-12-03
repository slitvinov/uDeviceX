#!/usr/bin/awk -f

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

    if (match(line, "^" identifier) ||
	match(line, /^./)) {                    # everything else
	tok = substr(line, 1, RLENGTH)
	line = substr(line, RLENGTH+1)
	return tok
    }
}

BEGIN {
    identifier = "[A-Za-z_][A-Za-z_0-9]*"
    asplit("void __global__ __device__ float float3 float2 int int4 bool double" \
	   " __forceinline__ const", wrds)
    
    for (;;) {
	advance()
	if (tok == "(eof)") break
	if (!(tok in wrds))
	    ans = ans tok
    }
    print ans
}
