#include <iostream>
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
	printf("Number of splits = %d \n", opt.nsplits);
	printf("Randomize vertex IDs = %d \n", opt.randomizeID);
  return;
}

void printHelp(const char* argv0) {
  printf("Usage: %s [options] <input mtx file> <output mtx file> \n", argv0);
	printf("Options:\n");
	printf("\t--help Print help message and exit.\n");

	printf("\t--selfloops\n");
	printf("\t\t0: Remove all self loops (default)\n");
	printf("\t\t1: Retain self loops\n");
	
	printf("\t--duplicatededges\n");
	printf("\t\t0: Remove all duplicated edges (default)\n");
	printf("\t\t1: Retain duplicated edges\n");
	
	printf("\t--uppertriangular\tAll edges (u,v) have u <= v \n");
	printf("\t--bidirectional\tFor all edges (u,v) add (v,u)\n");

	printf("\t--inputformat\n");
	printf("\t\t0: Binary input\n");
	printf("\t\t1: Text input (default)\n");
	
	printf("\t--outputformat\n");
	printf("\t\t0: Binary output (default)\n");
	printf("\t\t1: Text output\n");

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
	
	printf("\t--split [number] (default 1 i.e. no splitting, if specified then output file is split into n pieces each named as <outputfile>i_n for i from 0 to n-1)\n");
	printf("\t--randomizeID\tUsing this flag would randomize the vertex IDs from the input file and also produce a <output>.permutation file with the random permutation used\n");
}

template <typename T> struct edge {
	int src;
	int dst;
	T val;
};

template <typename T> void remove_selfloops(edge<T>*& edges, unsigned long int& nnz) {
	edge<T>* e2 = new edge<T>[nnz];
	unsigned long int nnz2 = 0;
	for(unsigned long int i = 0; i < nnz; i++) {
		if (edges[i].src != edges[i].dst) {
			e2[nnz2] = edges[i];
			nnz2++;
		}
	}
	delete [] edges;
	edges = e2;
	nnz = nnz2;	
	printf("%lu edges after removing self loops\n", nnz);
	return;
}

bool compare_for_duplicates_uint(const edge<unsigned int>& e1, const edge<unsigned int>& e2) {
	if (e1.src < e2.src) return true;
	else if (e1.src > e2.src) return false;
	if (e1.dst < e2.dst) return true;
  else return false;
}

void sort_types(edge<unsigned int>*& edges, unsigned long int nnz)
{
  __gnu_parallel::sort(edges, edges+nnz, compare_for_duplicates_uint);
}

bool compare_for_duplicates_double(const edge<double>& e1, const edge<double>& e2) {
	if (e1.src < e2.src) return true;
	else if (e1.src > e2.src) return false;
	if (e1.dst < e2.dst) return true;
  else return false;
}

void sort_types(edge<double>*& edges, unsigned long int nnz)
{
  __gnu_parallel::sort(edges, edges+nnz, compare_for_duplicates_double);
}

bool compare_for_duplicates_float(const edge<float>& e1, const edge<float>& e2) {
	if (e1.src < e2.src) return true;
	else if (e1.src > e2.src) return false;
	if (e1.dst < e2.dst) return true;
  else return false;
}

void sort_types(edge<float>*& edges, unsigned long int nnz)
{
  __gnu_parallel::sort(edges, edges+nnz, compare_for_duplicates_float);
}


template <typename T> void remove_duplicatededges(edge<T>*& edges, unsigned long int& nnz) {
	//sort by src & dst
	sort_types(edges, nnz);
	edge<T>* e2 = new edge<T>[nnz];
	unsigned long int nnz2 = 0;
	e2[0] = edges[0]; nnz2=1;

	for(unsigned long int i = 1; i < nnz; i++) {
		if ((edges[i].src == edges[i-1].src) && (edges[i].dst == edges[i-1].dst)) {
			continue;
		} else {
			e2[nnz2] = edges[i];
			nnz2++;
		}
	}
	delete [] edges;
	edges = e2;
	nnz = nnz2;	
	printf("%lu edges after removing duplicated edges\n", nnz);
	return;
}

template <typename T> void uppertriangular(edge<T>*& edges, unsigned long int& nnz) {
	for (unsigned long int i = 0; i < nnz; i++) {
		if (edges[i].src > edges[i].dst) {
			std::swap(edges[i].src, edges[i].dst);
		}
	}
	printf("%lu edges after retaining only upper triangular edges\n", nnz);
	return;
}

template <typename T> void bidirectional(edge<T>*& edges, unsigned long int& nnz) {
	edge<T>* e2 = new edge<T>[nnz*2];
	for (unsigned long int i = 0; i < nnz; i++) {
		e2[2*i] = edges[i];
		e2[2*i+1] = edges[i];
		std::swap(e2[2*i+1].src, e2[2*i+1].dst);
	}
	delete [] edges;
	edges = e2;
	nnz = 2*nnz;
	printf("%lu edges after adding bidirectional edges\n", nnz);
	return;
}

void readLine(FILE * ifile, int inputformat, int * src, int * dst, unsigned int * val, struct myoptions opt)
{
  if(inputformat == 0) {	
    fread(src, sizeof(int), 1, ifile);
    fread(dst, sizeof(int), 1, ifile);
    if (opt.inputedgeweights == 1) fread(val, sizeof(unsigned int), 1, ifile);
  } else {
    if (opt.inputedgeweights == 1) {
      fscanf(ifile, "%d %d %u", src, dst, val);
    } else {
      fscanf(ifile, "%d %d", src, dst);
    }
  }
}

void readLine (FILE * ifile, int inputformat, int * src, int * dst, double * val, struct myoptions opt)
{
  if(inputformat == 0) {	
    fread(src, sizeof(int), 1, ifile);
    fread(dst, sizeof(int), 1, ifile);
    if (opt.inputedgeweights == 1) fread(val, sizeof(double), 1, ifile);
  } else {
    if (opt.inputedgeweights == 1) {
      fscanf(ifile, "%d %d %lf", src, dst, val);
    } else {
      fscanf(ifile, "%d %d", src, dst);
    }
  }
}

void readLine (FILE * ifile, int inputformat, int * src, int * dst, float * val, struct myoptions opt)
{
  if(inputformat == 0) {	
    fread(src, sizeof(int), 1, ifile);
    fread(dst, sizeof(int), 1, ifile);
    if (opt.inputedgeweights == 1) fread(val, sizeof(float), 1, ifile);
  } else {
    if (opt.inputedgeweights == 1) {
      fscanf(ifile, "%d %d %f", src, dst, val);
    } else {
      fscanf(ifile, "%d %d", src, dst);
    }
  }
}

template <typename T> void readFile(const char* ifilename, edge<T>*& edges, int& n, unsigned long int& nnz, struct myoptions opt) {
	//if (opt.inputedgeweights == 0) {
	//	printf("Reader not implemented for no edge weights\n");
	//	exit(1);
	//}

	FILE * ifile;
	if (opt.inputformat == 0) {
		ifile = fopen(ifilename, "rb");
	} else {
		ifile = fopen(ifilename, "r");
	}

	assert(ifile != NULL);

	int m;
	unsigned int _nnz;
	if (opt.inputheader) {
		if (opt.inputformat == 0) {
			fread(&m, sizeof(int), 1, ifile);
			fread(&n, sizeof(int), 1, ifile);
			fread(&_nnz, sizeof(int), 1, ifile);
		} else {
			fscanf(ifile, "%d", &m);
			fscanf(ifile, "%d", &n);
			fscanf(ifile, "%d", &_nnz);
		}
		printf("Header found %d %d %lu in file %s\n", m, n, (unsigned long int)_nnz, ifilename);
		assert(m==n);

		if (opt.nvertices != 0) {
			//assert(opt.nvertices == n);
			printf("Warning: File had %d vertices, overriding with %d vertices\n", n, opt.nvertices);
			n = opt.nvertices;
		}
		nnz = _nnz;
	} else {
		n = 0; nnz = 0;
		int tempsrc, tempdst;
		T tempval;
		while(!feof(ifile)) {
		        readLine(ifile, opt.inputformat, &tempsrc, &tempdst, &tempval, opt);
			n = (n > tempsrc)?(n):(tempsrc);
			n = (n > tempdst)?(n):(tempdst);
			nnz++;
		}
		fseek(ifile, 0, SEEK_SET);
		if (opt.nvertices != 0) {
			assert(opt.nvertices >= n);
			n = opt.nvertices;	
		}
		//printf("Reader not implemented \n");
		//exit(1);

	} 

	edges = new edge<T>[nnz];
	//if (opt.inputformat == 0) {
	//	fread(edges, sizeof(edge<T>), (unsigned long int)nnz, ifile);
	//} else {
		for (unsigned long int i = 0; i < nnz; i++) {
			if (feof(ifile)) {
				printf("FEOF reached when reading edge id %lld\n", i);
			}
		        assert(!feof(ifile));
		        readLine(ifile, opt.inputformat, &edges[i].src, &edges[i].dst, &edges[i].val, opt);
		}
	//}
	printf("Read %lu edges and %d vertices from file %s\n", nnz, n, ifilename);
  
	fclose(ifile);
}

void writeLine (FILE * ofile, int src, int dst, unsigned int val)
{
  fprintf(ofile, "%d %d %d\n", src, dst, val);
}

void writeLine (FILE * ofile, int src, int dst, double val)
{
  fprintf(ofile, "%d %d %1.16e\n", src, dst, val);
}

void writeLine (FILE * ofile, int src, int dst, float val)
{
  fprintf(ofile, "%d %d %f\n", src, dst, val);
}

void writeLine (FILE * ofile, int src, int dst)
{
  fprintf(ofile, "%d %d\n", src, dst);
}

template <typename T> void writeFile(const char* ofilename, edge<T>* edges, int n, unsigned long int nnz, struct myoptions opt) {
	FILE * ofile;
	if (opt.outputformat == 0) {
		ofile = fopen(ofilename, "wb");
	} else {
		ofile = fopen(ofilename, "w");
	}

	assert(ofile != NULL);
	unsigned int _nnz = nnz;

	if (opt.outputheader) {
		if (opt.outputformat == 0) {
			fwrite(&n, sizeof(int), 1, ofile);
			fwrite(&n, sizeof(int), 1, ofile);
			fwrite(&_nnz, sizeof(int), 1, ofile);
		} else {
			fprintf(ofile, "%d %d %u\n", n, n, _nnz);
		}
	} else {
		//no header
	} 

        if (opt.outputedgeweights == 2) {
	  for (unsigned long int i = 0; i < nnz; i++) {
            edges[i].val = 1;  
          }
        }
        if (opt.outputedgeweights == 3) {
	  for (unsigned long int i = 0; i < nnz; i++) {
            double t = ((double)rand()/RAND_MAX*opt.random_range);  
            if (t > opt.random_range) t = opt.random_range;
            if (t < 0) t = 0;
            edges[i].val = (T) t;  
          }
        }

	if (opt.outputformat == 0) {
		if (opt.outputedgeweights == 0) {
			printf("Writer not yet implemented for no edge weights\n");
			exit(1);
		}
		//printf("First line %d %d %d\n", edges[0].src, edges[0].dst, edges[0].val);
		fwrite(edges, sizeof(edge<T>), nnz, ofile);
	} else {
		for (unsigned long int i = 0; i < nnz; i++) {
			if (opt.outputedgeweights == 0) {
                        	writeLine(ofile, edges[i].src, edges[i].dst);
			} else {
                        	writeLine(ofile, edges[i].src, edges[i].dst, edges[i].val);
			}
		}
	}
	printf("Wrote %lu edges and %d vertices to file %s\n", nnz, n, ofilename);
  
	fclose(ofile);
}

template<typename T> void process_graph(const char * ifilename, const char * ofilename, struct myoptions Opt)
{
	edge<T>* edges;
	int n;
	unsigned long int nnz;
	readFile<T>(ifilename, edges, n, nnz, Opt);

	if (Opt.selfloops == 0) {
		remove_selfloops(edges, nnz);
	}
	if (Opt.bidirectional == 1) {
		bidirectional(edges, nnz);
	}
	if (Opt.uppertriangular == 1) {
		uppertriangular(edges, nnz);
	}
	if (Opt.duplicatededges == 0) {
		remove_duplicatededges(edges, nnz);
	}
        if (Opt.randomizeID == 1) {
          int* perm = new int[n];
          std::iota(perm, perm+n, 1);
          std::random_device d;
          std::mt19937 g(d());
          std::shuffle(perm, perm+n, g);
 
          std::string permFile = ofilename + std::string(".permutation");
          std::cout << "Writing permutation to file " << permFile << std::endl;
          FILE* ofile = fopen(permFile.c_str(), "w");
          assert(ofile != NULL);
          for (int i = 0; i < n; i++) {
            fprintf(ofile, "%d %d \n", i+1, perm[i]);
          }
          fclose(ofile);

          #pragma omp parallel for
          for (int i = 0; i < nnz; i++) {
            edges[i].src = perm[edges[i].src - 1];
            edges[i].dst = perm[edges[i].dst - 1];
          }

          delete [] perm;
        }

	if (Opt.nsplits == 1) {
		writeFile(ofilename, edges, n, nnz, Opt);
	} else {
		unsigned long int nnz_per_file = nnz/Opt.nsplits;
		int remainder = nnz%Opt.nsplits;
		unsigned long int edge_offset = 0;
		for (int i = 0; i < Opt.nsplits; i++) {
			std::string ofilename_i =	std::string(ofilename) + 
							std::to_string((unsigned long long)i);
			unsigned long int nnz_written = nnz_per_file + ((remainder>0)?(1):(0)); 
			remainder--;
			writeFile<T>(ofilename_i.c_str(), edges + edge_offset, n, nnz_written, Opt);
			edge_offset += nnz_written;
		}
		assert(edge_offset == nnz);
	}
	delete [] edges;

/*	FILE* ifile = fopen(ifilename, "r");
	//read into array
	//
	
	fclose(ifile);

  //process according to options
  //

  FILE* ofile = fopen(ofilename, "wb");
	//write
	//
	fclose(ofile);	
*/


}

int main(int argc, char* argv[]) {

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

  return 0;
}
