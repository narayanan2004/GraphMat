#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <parallel/algorithm>
#include <utility>
#include <numeric>
#include <random>
#include "assert.h"
#include <getopt.h>

#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/binary_oarchive.hpp"
#include "GMDP/gmdp.h"
#include "Graph.h"

struct myoptions {
	int selfloops;
	int duplicatededges;
	int uppertriangular;
	int bidirectional;
	int inputformat;
	int outputformat;
	int inputheader;
	int outputheader;
	int inputedgeweights;
	int outputedgeweights;
	int edgeweighttype;
	int nvertices;
	int random_range;
	int nsplits;
        int randomizeID;
}; 

int validateOptions(struct myoptions opt) {
	int retval = 1;
	if (opt.selfloops != 0 && opt.selfloops != 1) {
		printf("selfloops must be 0 or 1 \n");
		retval = 0;
	}
	if (opt.uppertriangular == 1 && opt.bidirectional == 1) {
		printf("Cannot be both uppertriangular and bidirectional\n");
		retval = 0;
	}
	if (opt.inputedgeweights==0 && opt.outputedgeweights == 1) {
		printf("No input edge weights and want output edge weights\n");
		retval = 0;
	}
	if (opt.nsplits < 0) {
		printf("Cannot split into negative number of pieces\n");
		retval = 0;
	}
	if (opt.nsplits != 1) {
		printf("Split functionality is deprecated. Call with \"mpirun -np <nsplits> ... \" instead\n");
		retval = 0;
	}
	if (retval == 0) {
		printf("Error in validating options\n");
	}
	return retval;
}

struct myoptions initOptions() {
	struct myoptions opt;
	opt.selfloops = 0;
	opt.duplicatededges = 0;
	opt.uppertriangular = 0;
	opt.bidirectional = 0;
	opt.inputformat = 1;
	opt.outputformat = 0;
	opt.inputheader = 1;
	opt.outputheader = 1;
	opt.inputedgeweights = 1;	
	opt.outputedgeweights = 1;	
	opt.edgeweighttype = 0;	
	opt.nvertices = 0;
	opt.random_range = 128;
	opt.nsplits = 1;
        opt.randomizeID = 0;

	return opt;
}

void printOptions(struct myoptions opt) {
	printf("Options -- \n");
	printf("Selfloops = %d \n", opt.selfloops);
	printf("Duplicated edges = %d \n", opt.duplicatededges);
	printf("Uppertriangular = %d \n", opt.uppertriangular);
	printf("Bidirectional = %d \n", opt.bidirectional);
	printf("Input format = %d \n", opt.inputformat);
	printf("Output format = %d \n", opt.outputformat);
	printf("Input header = %d \n", opt.inputheader);
	printf("Output header = %d \n", opt.outputheader);
	printf("Input edge weights = %d \n", opt.inputedgeweights);
	printf("Output edge weights = %d \n", opt.outputedgeweights);
	printf("Edge weight type = %d \n", opt.edgeweighttype);
	printf("Range of random edge weights = %d \n", opt.random_range);
	printf("Number of vertices = %d \n", opt.nvertices);
	printf("Randomize vertex IDs = %d \n", opt.randomizeID);
  return;
}

void printHelp(const char* argv0) {
  printf("Usage: mpirun -np <nranks> %s [options] <input mtx file prefix> <output mtx file prefix> \n", argv0);
	printf("Options:\n");
	printf("\t--help Print help message and exit.\n");

	printf("\t--selfloops\n");
	printf("\t\t0: Remove all self loops (default)\n");
	printf("\t\t1: Retain self loops\n");
	
	printf("\t--duplicatededges\n");
	printf("\t\t0: Remove all duplicated edges (default)\n");
	printf("\t\t1: Retain duplicated edges\n");
	
	printf("\t--uppertriangular\tAll edges (u,v), leave edge unchanged if u <= v, and swap u & v if u > v\n");
	printf("\t--bidirectional\tFor all edges (u,v), add (v,u)\n");

	printf("\t--inputformat\n");
	printf("\t\t0: Binary mtx input\n");
	printf("\t\t1: Text mtx input (default)\n");
	printf("\t\t2: GraphMat format v2 (fast, but specialized to number of mpi ranks) \n");
	
	printf("\t--outputformat\n");
	printf("\t\t0: Binary mtx output (default)\n");
	printf("\t\t1: Text mtx output\n");
	printf("\t\t2: GraphMat format v2 (fast, but specialized to number of mpi ranks) \n");

	printf("\t--inputheader\n");
	printf("\t\t0: no header (can provide nvertices through --nvertices or we take the max as nvertices)\n");
	printf("\t\t1: (n,n,nnz) (default)\n");
	
	printf("\t--outputheader\n");
	printf("\t\t0: no header\n");
	printf("\t\t1: (n,n,nnz) (default)\n");

	printf("\t--inputedgeweights\n");
	printf("\t\t0: no weights\n");
	printf("\t\t1: weights present (default)\n");
	
	printf("\t--outputedgeweights\n");
	printf("\t\t0: no weights\n");
	printf("\t\t1: weights present (default)\n");
	printf("\t\t2: create unit weights\n");
	printf("\t\t3: create random weights in range [1,r) (specify r with --r option, default r=128) \n");

	printf("\t--edgeweighttype\n");
	printf("\t\t0: int (default)\n");
	printf("\t\t1: double\n");
	printf("\t\t2: float\n");
	
	printf("\t--r [number] range of random edge weights created (use only with \"--outputedgeweights 3\")\n");

	printf("\t--nvertices [number] (use only with \"--inputheader 0\")\n");
	
	printf("\t--randomizeID\tUsing this flag would randomize the vertex IDs from the input file\n");// and also produce a <output>.permutation file with the random permutation used\n");
}

template<typename T> 
void process_graph(const char * ifilename, const char * ofilename, struct myoptions Opt)
{
	//edge<T>* edges;
	GraphMat::edgelist_t<T> edgelist;
	int n;
	unsigned long int nnz;
	//readFile<T>(ifilename, edges, n, nnz, Opt);
	if (Opt.inputformat == 0 || Opt.inputformat==1) {
          bool binaryformat = (Opt.inputformat == 0);
          bool header = (Opt.inputheader == 1);
          bool edgeweights = (Opt.inputedgeweights == 1);
	  GraphMat::load_edgelist<T>(ifilename, &edgelist, binaryformat, header, edgeweights);
          auto maxn = std::max(edgelist.n, edgelist.m);
          edgelist.m = maxn;
          edgelist.n = maxn;
	  if (Opt.nvertices > 0) {
            assert(Opt.nvertices >= edgelist.m);
            assert(Opt.nvertices >= edgelist.n);
            edgelist.n = Opt.nvertices;
            edgelist.m = Opt.nvertices;
          }
        }
	if (Opt.inputformat == 2) {
		GraphMat::Graph<int, T> G;
		G.ReadGraphMatBin(ifilename);
		G.getEdgelist(edgelist);
	}
	if (Opt.outputedgeweights == 3) {
		GraphMat::random_edge_weights(&edgelist, Opt.random_range);
	}
	GraphMat::shuffle_edges(&edgelist);

	if (Opt.selfloops == 0) {
		GraphMat::remove_selfedges(&edgelist);
	}
	if (Opt.bidirectional == 1) {
		GraphMat::create_bidirectional_edges(&edgelist);
	}
	if (Opt.uppertriangular == 1) {
		GraphMat::convert_to_dag(&edgelist);
	}
	if (Opt.duplicatededges == 0) {
		GraphMat::remove_duplicate_edges(&edgelist);
	}
        if (Opt.randomizeID == 1) {
        	GraphMat::randomize_edgelist_square(&edgelist);
        }
	if (Opt.outputformat == 0 || Opt.outputformat==1) {
          bool binaryformat = (Opt.outputformat == 0);
          bool header = (Opt.outputheader == 1);
          bool edgeweights = (Opt.outputedgeweights != 0);
	  GraphMat::write_edgelist<T>(ofilename, edgelist, binaryformat, header, edgeweights);
	}
	if (Opt.outputformat == 2) {
		GraphMat::Graph<int, T> G;
		G.ReadEdgelist(edgelist);
		G.WriteGraphMatBin(ofilename);
	}

	edgelist.clear();
}

int main(int argc, char* argv[]) {

	MPI_Init(&argc, &argv);
	struct myoptions Opt = initOptions();
	static struct option longOptions[] = {
      {"uppertriangular", no_argument, &Opt.uppertriangular, 1},
      {"bidirectional", no_argument, &Opt.bidirectional, 1},
      {"selfloops", required_argument, 0, 's'},
      {"duplicatededges", required_argument, 0, 'd'},
      {"inputformat", required_argument, 0, 'i'},
      {"outputformat", required_argument, 0, 'o'},
      {"inputheader", required_argument, 0, 'n'},
      {"outputheader", required_argument, 0, 'u'},
      {"inputedgeweights", required_argument, 0, 'e'},
      {"outputedgeweights", required_argument, 0, 'w'},
      {"edgeweighttype", required_argument, 0, 't'},
      {"r", required_argument, 0, 'r'},
      {"nvertices", required_argument, 0, 'v'},
      {"split", required_argument, 0, 'p'},
      {"randomizeID", no_argument, &Opt.randomizeID, 1},
      {"help", no_argument, 0, 'h'}
    };

  while (1) {
    int optionIndex = 0;
    int currentOption = getopt_long(argc, (char *const*)argv, "h:s:d:i:o:n:u:e:w:t:v:r:p", longOptions, &optionIndex);
    if (currentOption == -1) {
      break;
    }
    int method = 3;
    switch (currentOption) {
    case 0:
      break;
    case 'h':
      printHelp(argv[0]);
      return(0);
    case 's':
      sscanf(optarg, "%i", &Opt.selfloops);
      break;
    case 'd':
      sscanf(optarg, "%i", &Opt.duplicatededges);
      break;
    case 'i':
      sscanf(optarg, "%i", &Opt.inputformat);
      break;
    case 'o':
      sscanf(optarg, "%i", &Opt.outputformat);
      break;
    case 'n':
      sscanf(optarg, "%i", &Opt.inputheader);
      break;
    case 'u':
      sscanf(optarg, "%i", &Opt.outputheader);
      break;
    case 'e':
      sscanf(optarg, "%i", &Opt.inputedgeweights);
      break;
    case 'w':
      sscanf(optarg, "%i", &Opt.outputedgeweights);
      break;
    case 't':
      sscanf(optarg, "%i", &Opt.edgeweighttype);
      break;
    case 'v':
      sscanf(optarg, "%i", &Opt.nvertices);
      break;
    case 'r':
      sscanf(optarg, "%i", &Opt.random_range);
      break;
    case 'p':
      sscanf(optarg, "%i", &Opt.nsplits);
      break;

    case '?':
      break;
    default:
      abort();
      break;
    }
  }

  if (optind != argc - 2) {
    printHelp(argv[0]);
    return(0);
	}

	int check = validateOptions(Opt);
	assert(check != 0);

  printOptions(Opt);

  const char* ifilename = argv[optind];
  const char* ofilename = argv[optind+1];

  switch(Opt.edgeweighttype)
  {
    case 0:
      process_graph<unsigned int>(ifilename, ofilename, Opt);
      break;
    case 1:
      process_graph<double>(ifilename, ofilename, Opt);
      break;
    case 2:
      process_graph<float>(ifilename, ofilename, Opt);
      break;
    default:
      printf("Invalid edge type: %d\n", Opt.edgeweighttype);
      exit(-1);
      break;
  }

  MPI_Finalize();
  return 0;
}
