#!/usr/bin/awk -f

# Write every function definition into a separate file Note: start and
# end of a function must be tagged (see data/rbc-cuda.taged.cu)

function fbegp() {
    return substr($0, 1, 5) == "/*s*/"
}

function fendp() {
    return substr($0, 1, 5) == "/*e*/"    
}

function untag() { # remove tag 
    sub(/^[\/][*][^*]+[*][\/]/, "")
}

function remword(w, s) {
    gsub("[\t ]+" w " ", " ", s)
    gsub("^" w " ", " ", s)
    return s
}

function fname(s,    w) { # return function name
    sub("[(].*", "", s) # delete (...
    for (w in wrds)
	s = remword(w, s)
    gsub(" ", "", s)
    return s
}

function asplit(str, arr,    temp) {  # make an assoc array from str
    n = split(str, temp)
    for (i = 1; i <= n; i++)
        arr[temp[i]]++
    return n
}

BEGIN {
    asplit("void __global__ __device__ float float3 float2 int bool double inline" \
	   " __forceinline__", wrds)
    
    d = "rbc-cuda"
    cmd = sprintf("mkdir -p \"%s\"", d)
    system(cmd)
}

fbegp() {
    untag()
    fn = d "/" fname($0) ".cpp"

    printf "(fsplit.awk) writing: %s\n" , fn > "/dev/stderr"
    
    for(;;) {
	untag()
	print > fn
	if (getline <= 0) break
	if (fendp())      {
	    untag()
	    print > fn
	    break
	}
    }
    close(fn)
}
