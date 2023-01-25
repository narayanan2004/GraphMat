MPICXX=mpiicpc
CXX=dpcpp

SRCDIR=./src
INCLUDEDIR=./include
DIST_PRIMITIVES_PATH=$(INCLUDEDIR)/GMDP
BINDIR=./bin

CATCHDIR=./test/Catch
TESTDIR=./test
TESTBINDIR=./testbin


ifeq (${CXX}, dpcpp)
	CXX_OPTIONS=-qopenmp -std=c++11 -I/opt/intel/oneapi/mpi/latest/include
else
	CXX_OPTIONS=-fopenmp --std=c++11 -I/usr/include/mpi/
endif

CXX_OPTIONS+=-I$(INCLUDEDIR) -I$(DIST_PRIMITIVES_PATH) -I${BOOST_ROOT}

ifeq (${debug}, 1)
	CXX_OPTIONS += -O0 -g -D__DEBUG 
else
	ifeq (${CXX}, dpcpp)
	  CXX_OPTIONS += -O3 -ipo 
	else
	  CXX_OPTIONS += -O3 -flto -fwhole-program
	endif
endif

ifeq (${CXX}, dpcpp)
	CXX_OPTIONS += -xHost
else
	CXX_OPTIONS += -march=native
endif

ifeq (${timing}, 1)
	CXX_OPTIONS += -D__TIMING
else
endif

CXX_OPTIONS += -Wno-format -Wno-dangling-else

LD_OPTIONS += -L${BOOST_ROOT}/stage/lib/ -lboost_serialization

# --- Apps --- #
sources = $(wildcard $(SRCDIR)/*.cpp)
include_headers = $(wildcard $(INCLUDEDIR)/*.h)
dist_primitives_headers = $(wildcard $(DIST_PRIMITIVES_PATH)/*.h $(DIST_PRIMITIVES_PATH)/*/*.h)
deps = $(include_headers) $(dist_primitives_headers)
apps = $(patsubst $(SRCDIR)/%.cpp, $(BINDIR)/%, $(sources))

all: $(apps)

$(BINDIR)/% : $(SRCDIR)/%.cpp $(deps)  
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -o $@ $< $(LD_OPTIONS)

# --- Test --- #
test: $(TESTBINDIR)/test 
test_headers = $(wildcard $(TESTDIR)/*.h)
test_src = $(wildcard $(TESTDIR)/*.cpp)
test_objects = $(patsubst $(TESTDIR)/%.cpp, $(TESTBINDIR)/%.o, $(test_src))

$(TESTBINDIR)/%.o : $(TESTDIR)/%.cpp $(deps) $(test_headers) 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -I$(CATCHDIR)/include -c $< -o $@ $(LD_OPTIONS)

$(TESTBINDIR)/test: $(test_objects) 
	$(MPICXX) -cxx=$(CXX) $(CXX_OPTIONS) -I$(CATCHDIR)/include -o $(TESTBINDIR)/test $(test_objects) $(LD_OPTIONS)

# --- clean --- #

clean:
	rm -f $(apps) $(TESTBINDIR)/test $(test_objects)
