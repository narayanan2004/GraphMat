
CATCHDIR=./test/Catch
TESTDIR=./test
TESTBINDIR=./testbin

MPICXX=mpiicpc
CXX=icpc

ifeq (${CXX}, icpc)
  CXX_OPTIONS=-qopenmp -std=c++11 
else
  CXX_OPTIONS=-fopenmp --std=c++11 -I/usr/include/mpi/
endif

CXX_OPTIONS+=-Isrc


ifeq (${debug}, 1)
  CXX_OPTIONS += -O0 -g -D__DEBUG 
else
  ifeq (${CXX}, icpc)
    CXX_OPTIONS += -O3 -ipo 
  else
    CXX_OPTIONS += -O3 -flto -fwhole-program
  endif
endif

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
DIST_PRIMITIVES_PATH=$(SRCDIR)/GMDP
BINDIR=bin

SOURCES=$(SRCDIR)/PageRank.cpp $(SRCDIR)/Degree.cpp $(SRCDIR)/BFS.cpp $(SRCDIR)/SGD.cpp $(SRCDIR)/TriangleCounting.cpp $(SRCDIR)/SSSP.cpp $(SRCDIR)/Delta.cpp

DEPS=$(SRCDIR)/SPMV.cpp $(SRCDIR)/Graph.cpp $(SRCDIR)/GraphProgram.cpp $(SRCDIR)/GraphMatRuntime.cpp $(DIST_PRIMITIVES_PATH)/matrices/layouts.h $(DIST_PRIMITIVES_PATH)/gmdp.h

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

# --- Test --- #
test: $(TESTBINDIR)/test 
test_headers = $(wildcard $(TESTDIR)/*.hpp)
test_src = $(wildcard $(TESTDIR)/*.cpp)
test_objects = $(patsubst $(TESTDIR)/%.cpp, $(TESTBINDIR)/%.o, $(test_src))

$(TESTBINDIR)/%.o : $(TESTDIR)/%.cpp $(DEPS) $(test_headers) 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -I$(CATCHDIR)/include $(CXX_OPTIONS) -c $< -o $@

$(TESTBINDIR)/test: $(test_objects) 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -I$(CATCHDIR)/include $(CXX_OPTIONS) -o $(TESTBINDIR)/test $(test_objects)

# --- clean --- #

clean:
	rm $(EXE) bin/graph_converter $(TESTBINDIR)/test $(test_objects)
