
GRAPHBLAS_PATH=../PCLGraphBLAS/
include $(GRAPHBLAS_PATH)/Make.inc

#CXX=icpc
CXX=mpiicpc
CXX_OPTIONS=-openmp -std=c++11 -I./src/ -I$(GRAPHBLAS_PATH)


ifeq (${debug}, 1)
  CXX_OPTIONS += -O0 -g -D__DEBUG $(GBFLAGS)
else
  CXX_OPTIONS += -O3 -ipo $(GBFLAGS)
endif

CXX_OPTIONS += -xHost

ifeq (${timing}, 1)
  CXX_OPTIONS += -D__TIMING
else
endif

SRCDIR=src
BINDIR=bin

SOURCES=$(SRCDIR)/PageRank.cpp $(SRCDIR)/Degree.cpp $(SRCDIR)/BFS.cpp $(SRCDIR)/SGD.cpp $(SRCDIR)/TriangleCounting.cpp $(SRCDIR)/SSSP.cpp $(SRCDIR)/Delta.cpp

DEPS=$(SRCDIR)/SPMV.cpp $(SRCDIR)/Graph.cpp $(SRCDIR)/GraphProgram.cpp $(SRCDIR)/SparseVector.cpp $(SRCDIR)/GraphMatRuntime.cpp $(GRAPHBLAS_PATH)/src/layouts.h $(GRAPHBLAS_PATH)/src/graphpad.h

EXE=$(BINDIR)/PageRank $(BINDIR)/IncrementalPageRank $(BINDIR)/BFS $(BINDIR)/SSSP #$(BINDIR)/TriangleCounting $(BINDIR)/SGD $(BINDIR)/DS


all: $(EXE) graph_converter
	
#bin/layouts.o: $(GRAPHBLAS_PATH)/ipdps/layouts.cc $(GRAPHBLAS_PATH)/ipdps/layouts.h
#	$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/layouts.o -c $(GRAPHBLAS_PATH)/ipdps/layouts.cc
        
graph_converter: graph_utils/graph_convert.cpp
	$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/graph_converter graph_utils/graph_convert.cpp

$(BINDIR)/PageRank: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/PageRank.cpp $(SRCDIR)/Degree.cpp 
	$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/PageRank $(SRCDIR)/PageRank.cpp  

$(BINDIR)/IncrementalPageRank: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/IncrementalPageRank.cpp $(SRCDIR)/Degree.cpp 
	$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/IncrementalPageRank $(SRCDIR)/IncrementalPageRank.cpp 

$(BINDIR)/BFS: $(DEPS) $(SRCDIR)/BFS.cpp 
	$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/BFS $(SRCDIR)/BFS.cpp 

$(BINDIR)/SGD: $(DEPS) $(SRCDIR)/SGD.cpp 
	$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/SGD $(SRCDIR)/SGD.cpp

$(BINDIR)/TriangleCounting: $(DEPS) $(SRCDIR)/TriangleCounting.cpp
	$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/TriangleCounting $(SRCDIR)/TriangleCounting.cpp

$(BINDIR)/SSSP: $(DEPS) $(MULTINODEDEPS) $(SRCDIR)/SSSP.cpp 
	$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/SSSP $(SRCDIR)/SSSP.cpp 

$(BINDIR)/DS: $(DEPS) $(SRCDIR)/Delta.cpp
	$(CXX) $(CXX_OPTIONS) -o $(BINDIR)/DS $(SRCDIR)/Delta.cpp

clean:
	rm $(EXE) bin/graph_converter 
