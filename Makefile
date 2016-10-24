
DIST_PRIMITIVES_PATH=GraphMatDistributedPrimitives
CATCHDIR=./test/Catch
TESTDIR=./test
TESTBINDIR=./testbin
include $(DIST_PRIMITIVES_PATH)/Make.inc

MPICXX=mpiicpc
CXX=icpc

ifeq (${CXX}, icpc)
  CXX_OPTIONS=-qopenmp -std=c++11 
else
  CXX_OPTIONS=-fopenmp --std=c++11 -I/usr/include/mpi/
endif

CXX_OPTIONS+=-Isrc -I$(DIST_PRIMITIVES_PATH)


ifeq (${debug}, 1)
  CXX_OPTIONS += -O0 -g -D__DEBUG 
else
  ifeq (${CXX}, icpc)
    CXX_OPTIONS += -O3 -ipo 
  else
    CXX_OPTIONS += -O3 -flto -fwhole-program
  endif
endif

CXX_OPTIONS += $(GPFLAGS)

ifeq (${CXX}, icpc)
  CXX_OPTIONS += -xHost
else
  CXX_OPTIONS += -march=native
endif

ifeq (${timing}, 1)
  CXX_OPTIONS += -D__TIMING
else
endif

SRCDIR=src
BINDIR=bin

SOURCES=$(SRCDIR)/PageRank.cpp $(SRCDIR)/Degree.cpp $(SRCDIR)/BFS.cpp $(SRCDIR)/SGD.cpp $(SRCDIR)/TriangleCounting.cpp $(SRCDIR)/SSSP.cpp $(SRCDIR)/Delta.cpp

DEPS=$(SRCDIR)/SPMV.cpp $(SRCDIR)/Graph.cpp $(SRCDIR)/GraphProgram.cpp $(SRCDIR)/SparseVector.cpp $(SRCDIR)/GraphMatRuntime.cpp $(DIST_PRIMITIVES_PATH)/src/layouts.h $(DIST_PRIMITIVES_PATH)/src/graphpad.h

EXE=$(BINDIR)/PageRank $(BINDIR)/IncrementalPageRank $(BINDIR)/BFS $(BINDIR)/SSSP $(BINDIR)/LDA $(BINDIR)/SGD $(BINDIR)/TriangleCounting #$(BINDIR)/DS


all: $(EXE) graph_converter
	
graph_converter: graph_utils/graph_convert.cpp
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/graph_converter graph_utils/graph_convert.cpp

$(BINDIR)/PageRank: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/PageRank.cpp $(SRCDIR)/Degree.cpp 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/PageRank $(SRCDIR)/PageRank.cpp  

$(BINDIR)/IncrementalPageRank: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/IncrementalPageRank.cpp $(SRCDIR)/Degree.cpp 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/IncrementalPageRank $(SRCDIR)/IncrementalPageRank.cpp 

$(BINDIR)/BFS: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/BFS.cpp 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/BFS $(SRCDIR)/BFS.cpp 

$(BINDIR)/SGD: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/SGD.cpp 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/SGD $(SRCDIR)/SGD.cpp

$(BINDIR)/TriangleCounting: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/TriangleCounting.cpp
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/TriangleCounting $(SRCDIR)/TriangleCounting.cpp

$(BINDIR)/SSSP: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/SSSP.cpp 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/SSSP $(SRCDIR)/SSSP.cpp 

$(BINDIR)/LDA: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/LDA.cpp 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/LDA $(SRCDIR)/LDA.cpp 

$(BINDIR)/DS: $(DEPS) $(SRCDIR)/Delta.cpp
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/DS $(SRCDIR)/Delta.cpp

test: $(TESTBINDIR)/mat1

$(TESTBINDIR)/mat1: $(DEPS) $(TESTDIR)/mat1.cpp
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -I$(CATCHDIR)/include $(CXX_OPTIONS) -o $(TESTBINDIR)/mat1 $(TESTDIR)/mat1.cpp 

clean:
	rm $(EXE) bin/graph_converter  $(TESTBINDIR)/mat1
