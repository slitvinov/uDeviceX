#!/usr/bin/awk -f
#
# Testing framework
# See atest.example.sh for an example

function init () {
    VERBOSE=1
    EXIT_ON_FAIL=1

    # where the reference data for the tests are stored
    REF_DIR="test_data"
}

# format a error message
# v: verbosity level
# s: string to output
function msg(v, s) {
    if (v<=VERBOSE)
	printf "(atest.awk)%s\n", s > "/dev/stderr"
}

# extract test body
function test_body(        body, sep, rc, ns) {
    body = sep = ""
    do {
	rc = getline
	if (!rc)
	    return body
	ns = split($0, aux, "#")
	sub("[^#]*#", "")
	if (ns<2 || !NF)
	    return body
	body = body sep $0
	sep  = "\n"
    } while (1)
}

# set global variables tname, `body', `out_file', `ref_file'
function process_test_body() {
    tname = $2
    body = test_body()
    out_file = ext_outfile(body)
    if (!out_file)  {
	msg(0, "(ERROR) cannot find output file in test: " tname)
	msg(0, "        Script body:\n\n" body "\n\n")
	if (EXIT_ON_FAIL)
	    exit

    }

    
    ref_file = out2ref(out_file, tname)
}

# save a failed script to a file
function write_failed_script(dep_name) {
    dep_name = "fail." tname  ".sh"
    print body "\n" cmp_command "\n" > dep_name
    close(dep_name)
    msg(1, " Failed script is in " dep_name " \n")
}

# extract output file from test body
#  File should be in the form: *.out.*
#  Example:  ./run.sh > run.out.vtk
function ext_outfile(body) {
    match(body, /[[:alnum:]_]+\.out\.[[:alnum:]_\.]+/)
    return substr(body, RSTART, RLENGTH)
}

# Convert a name of an output file to a name of reference file
# Example: run.out.vtk -> <ref dir>/run.ref.<test name>.vtk
function out2ref(out_file, test_name) {
    sub(".out.", ".ref." test_name ".", out_file)
    return REF_DIR "/" out_file
}

BEGIN {
    init()
    system("mkdir -p " REF_DIR)
    ipass = ifail = icreated = 0
}

{
    ns = split($0, aux, "#")    
    # skip non-comments
    if (ns<2)
	next
    #$0 = aux[2]
    sub("[^#]*#", "")
}

# create test
$1 == "cTEST:" {
    process_test_body()
    sub(out_file, ref_file, body)
    msg(1, " CREATING: " tname)
    msg(1, "        Script body:\n\n" body "\n\n")
    rc = system(body)
    if (rc) {
	msg(0, " fail to create test\n")
	exit(-1)
    }
    # number of created scripts
    icreated++
}

# run test
$1 == "TEST:" {
    process_test_body()
    msg(1, " RUNNING: " tname)
    rc = system(body)
    if (rc) {
	ifail++
	msg(0, " fail to run test")
	msg(0, "        Test body:\n\n" body "\n")
	if (EXIT_ON_FAIL)
	    exit
	next
    }
    
    # compare output and reference files
    cmp_command = " cmp -s " out_file " " ref_file
    rc = system(cmp_command)
    if (!rc)  {
	ipass++
	msg(1, " PASSED : " tname)
    } else {
	ifail++
	msg(0, " FAILED : " tname)
	msg(0, " Test body:\n\n" body "\n" cmp_command "\n")
	write_failed_script()
	msg(0, " Files : `" out_file "` and `" ref_file "` are different")
	if (VERBOSE>=2)
	    system("diff " out_file " " ref_file)
	if (EXIT_ON_FAIL)
	    exit
    }
}

END {
    # output summary
    msg(0, sprintf(" %d/%d test(s) passed", ipass, ipass+ifail))
    if (icreated)
	msg(0, sprintf("   %d test(s) created", icreated))
}
