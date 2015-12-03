#!/usr/bin/awk -f

# Transform
# cpp functions from uDevice to maxima

function asplit(str, arr,    temp) {  # make an assoc array from str
    n = split(str, temp)
    for (i = 1; i <= n; i++)
        arr[temp[i]]++
    return n
}

function remword(w, s) {
    gsub("[\t ]+" w " ", " ", s)
    gsub("^" w " "     , " ", s)
    gsub("[(]" w " "   , "(", s)
    return s
}

function remwords(s,    rst, w) {
    for (w in wrds)
	s = remword(w, s)
    return s
}

BEGIN {
    asplit("void __global__ __device__ float float3 float2 int bool double" \
	   " __forceinline__ const", wrds)
}


function clear_line() {
    sub("//.*", "")      # remove comments
    
    $0 = remwords($0)

    $0 = replace_assignment($0)

    
    gsub("=", ":")
    gsub("{" , "(")
    gsub("}" , ")")
}

function process_head(   def) {
    clear_line()
    def = $0
    while ($0 !~ "[\)]") {
	getline
	clear_line()
	def = def $0
    }
    
    sub(/[)]/, "):=", def)
    print def
}

NR == 1 {
    process_head()

    next
}

{
    clear_line()
    print
}
