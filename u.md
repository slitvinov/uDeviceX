# [u]nits

uDeviceX consists of units. They are implemented in [/src/u/](/src/u/).  
See the [units](/doc/units/) folder for documentation of specific units.  

example:
* [x](/src/u/x/) "standard" udx unit
* [hw](/src/u/hw/) "Hello world!"

## Build and run

Compile unit `A`: from `/src`

	u.conf0 ./u/A
	u.make -j

## Update make file fragments

From `/src`

	u.u u/A

updates `hw/make/*.mk` files

## Create a new unit

From `src`

	mkdir -p hw/make

Add two file: `hw/make/i` and `hw/make/e`. The files are used by `u.u`
to create a list of unit source files. `i` is a script which returns a
list of [i]ncluded files. `e` returns a list of [e]xcluded files. The
`i` list "minus" `e` list is used as a source. `e` file is
optional. In other words `e` "black lists" files returned by `i`.

For `i` and `e` the variable `$U` is set to `u/hw`.

Run

	u.u u/hw

Add `u/hw/make/dep.mk u/hw/make/obj.mk u/hw/make/dir.mk
u/hw/make/rule.mk` to git.
