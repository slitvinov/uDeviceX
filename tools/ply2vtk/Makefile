CXXFLAGS=-O3 -Wall
LDLIBS=-lm
BIN = $(HOME)/bin

P= ply2vtk
all: $P

CXXFLAGS+=-std=c++11

install:; mkdir -p $(BIN) && cp $P $(BIN)

.PHONY: clean install
clean:; -rm -rf $P
