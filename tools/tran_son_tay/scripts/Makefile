BIN = $(HOME)/bin
PROG = post.py
UDX = $(HOME)/.udx
S = sphere.ply

install:
	mkdir -p $(BIN)
	cp $(PROG) $(BIN)
	python setup.py install --user
	mkdir -p $(UDX)
	cp $(S) $(UDX)

.PHONY: install
