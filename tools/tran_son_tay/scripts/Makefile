BIN = $(HOME)/bin
PROG = post.py fit.py get_sc.sh rename.sh skalak.py
UDX = $(HOME)/.udx
S = sphere.ply

install:
	mkdir -p $(BIN)
	cp $(PROG) $(BIN)
	python setup.py install --user
	mkdir -p $(UDX)
	cp $(S) $(UDX)

.PHONY: install
