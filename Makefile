MPICXX=mpiicpc
CXX=icpc

SRCDIR=./src
INCLUDEDIR=./include
DIST_PRIMITIVES_PATH=$(INCLUDEDIR)/GMDP
BINDIR=./bin

CATCHDIR=./test/Catch
TESTDIR=./test
TESTBINDIR=./testbin

ifeq (${CXX}, icpc)
  CXX_OPTIONS=-qopenmp -std=c++11
else
  CXX_OPTIONS=-fopenmp --std=c++11 -I/usr/include/mpi/
endif

CXX_OPTIONS+=-I$(INCLUDEDIR) -I$(DIST_PRIMITIVES_PATH)


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

LD_OPTIONS += -lboost_serialization

# --- Apps --- #
SOURCES = $(wildcard $(SRCDIR)/*.cpp)

include_headers = $(wildcard $(INCLUDEDIR)/*.h)
dist_primitives_headers = $(wildcard $(DIST_PRIMITIVES_PATH)/*.h $(DIST_PRIMITIVES_PATH)/*/*.h)
DEPS = $(include_headers) $(dist_primitives_headers)

APPS=$(BINDIR)/graph_converter $(BINDIR)/PageRank $(BINDIR)/IncrementalPageRank $(BINDIR)/BFS $(BINDIR)/SSSP $(BINDIR)/LDA $(BINDIR)/SGD $(BINDIR)/TriangleCounting #$(BINDIR)/DS

all: $(APPS)
	
$(BINDIR)/% : $(SRCDIR)/%.cpp $(DEPS)  
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $@ $< $(LD_OPTIONS)

# --- Test --- #
test: $(TESTBINDIR)/test 
test_headers = $(wildcard $(TESTDIR)/*.h)
test_src = $(wildcard $(TESTDIR)/*.cpp)
test_objects = $(patsubst $(TESTDIR)/%.cpp, $(TESTBINDIR)/%.o, $(test_src))

$(TESTBINDIR)/%.o : $(TESTDIR)/%.cpp $(DEPS) $(test_headers) 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -I$(CATCHDIR)/include -c $< -o $@ $(LD_OPTIONS)

$(TESTBINDIR)/test: $(test_objects) 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -I$(CATCHDIR)/include -o $(TESTBINDIR)/test $(test_objects) $(LD_OPTIONS)

# --- clean --- #

clean:
	rm -f $(APPS) $(TESTBINDIR)/test $(test_objects)
