/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ******************************************************************************/
/* Narayanan Sundaram (Intel Corp.), Michael Anderson (Intel Corp.)
 * ******************************************************************************/

#include <cstdalign>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <sys/time.h>
#include <parallel/algorithm>
#include <omp.h>
#include <cassert>

inline double sec(struct timeval start, struct timeval end)
{
    return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))))/1.0e6;
}

template <typename edge_value_type=int>
struct __attribute__((aligned(16))) edge_t
{
  int src;
  int dst;
  edge_value_type val;
  int partition_id;
};

template <typename edge_value_type=int>
struct edge_io_t
{
  int src;
  int dst;
  edge_value_type val;
};

extern int nthreads;

//const int MAX_PARTS = 512; 
template<class T>
void AddFn(T a, T b, T* c, void* vsp) {
  *c = a + b ;
}

template <class V, class E=int>
class Graph {

  public:
    V* vertexproperty;
    MatrixDC<E>** mat;
    MatrixDC<E>** matT; //transpose
    int nparts;
    //unsigned long long int* id; // vertex id's if any
    bool* active;
    int nvertices;
    long long int nnz;
    int vertexpropertyowner;

    int *start_src_vertices; //start and end of transpose parts
    int *end_src_vertices;
//    int start_dst_vertices[MAX_PARTS];
//    int end_dst_vertices[MAX_PARTS];

  public:
    void ReadMTX(const char* filename, int grid_size); 
    //void ReadMTX_old(const char* filename, int grid_size); 
    void ReadMTX_sort(const char* filename, int grid_size); 
    void ReadMTX_sort(edge_t<E>* edges, int m, int n, int nnz, int grid_size, int alloc=1); 
    void setAllActive();
    void setAllInactive();
    void setActive(int v);
    void setInactive(int v);
    void setAllVertexproperty(const V& val);
    void setVertexproperty(int v, const V& val);
    V getVertexproperty(int v) const;
    void reset();
    void shareVertexProperty(Graph<V,E>& g);
    int getBlockIdBySrc(int vertexid) const;
    int getBlockIdByDst(int vertexid) const;
    int getNumberOfVertices() const;
    void applyToAllVertices(void (*ApplyFn)(V, V*, void*), void* param=nullptr);
    template<class T> void applyReduceAllVertices(T* val, void (*ApplyFn)(V*, T*, void*), void (*ReduceFn)(T,T,T*,void*)=AddFn<T>, void* param=nullptr);
    ~Graph();
};

// int Read(char *fname, int*** _M, int **_Mg, int** _Mmg ,int* _g, int* m1) {
  // FILE* fp = fopen(fname, "r");
  // int m, n, nnz;
  // fscanf(fp, "%d %d %d", &m, &n, &nnz);
  // *m1 = m;

  // if (*_g > m) {
    // *_g = m;
  // }
  // int g = *_g;

  // *_M = (int**) _mm_malloc(sizeof(int*) * 3, 64);
  // *_Mg = (int*) _mm_malloc(sizeof(int) * (g + 1), 64);
  // *_Mmg = (int*) _mm_malloc(sizeof(int) * (g + 1), 64);

  // int** M = *_M;
  // int *Mmg = *_Mmg;
  // int *Mg = *_Mg;

  // M[0] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
  // M[1] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
  // M[2] = (int*) _mm_malloc(sizeof(int) * nnz, 64);

  // int r = m % g;
  // int q = m / g;

  // Mmg[0] = 1;
  // //printf("%d ", Mmg[0]);
  // for (int i = 1; i <= g; i++) {
    // if (r) {
      // Mmg[i] = Mmg[i - 1] + q + 1;
      // //printf("%d ", Mmg[i]);
      // r--;
    // } else {
      // Mmg[i] = Mmg[i - 1] + q;
      // //printf("%d ", Mmg[i]);
    // }
  // }
  // //printf("\n");

  // int tid = 0;
  // for (int i = 0; i < nnz; i++) {
    // fscanf(fp, "%d %d %d", &M[0][i], &M[1][i], &M[2][i]);
    // while (M[0][i] >= Mmg[tid]) {
      // Mg[tid] = i;
      // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
      // tid++;
    // }
  // }
  // Mg[tid] = nnz;
  // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
  // for (tid = tid + 1; tid <= g; tid++) {
    // Mg[tid] = nnz;
    // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
  // }
  // //printf("--------\n");
  // fclose(fp);

  // return n;
// }


// int ReadBinary(char *fname, int*** _M, int **_Mg, int** _Mmg ,int* _g, int* m1) {
  // FILE* fp = fopen(fname, "rb");
  // int m, n, nnz;
  // //fscanf(fp, "%d %d %d", &m, &n, &nnz);
  // int tmp_[3];
  // fread(tmp_, sizeof(int), 3, fp);
  // m = tmp_[0];
  // n = tmp_[1];
  // nnz = tmp_[2];
  // printf("Graph %d x %d : %d edges \n", m, m, nnz);

  // *m1 = m;

  // if (*_g > m) {
    // *_g = m;
  // }
  // int g = *_g;

  // *_M = (int**) _mm_malloc(sizeof(int*) * 3, 64);
  // *_Mg = (int*) _mm_malloc(sizeof(int) * (g + 1), 64);
  // *_Mmg = (int*) _mm_malloc(sizeof(int) * (g + 1), 64);

  // int** M = *_M;
  // int *Mmg = *_Mmg;
  // int *Mg = *_Mg;

  // size_t nnz_l = nnz;
  // M[0] = (int*) _mm_malloc(sizeof(int) * nnz_l, 64);
  // M[1] = (int*) _mm_malloc(sizeof(int) * nnz_l, 64);
  // M[2] = (int*) _mm_malloc(sizeof(int) * nnz_l, 64);

  // int* data_dump = (int*) _mm_malloc(sizeof(int)*nnz_l*3, 64);
  // fread(data_dump, sizeof(int), nnz_l*3, fp);

  // int r = m % g;
  // int q = m / g;
  // Mmg[0] = 0;
  // //printf("%d ", Mmg[0]);
  // for (int i = 1; i <= g; i++) {
    // if (r) {
      // Mmg[i] = Mmg[i - 1] + q + 1;
      // //printf("%d ", Mmg[i]);
      // r--;
    // } else {
      // Mmg[i] = Mmg[i - 1] + q;
      // //printf("%d ", Mmg[i]);
    // }
  // }
  // for (int i = 0; i <= g; i++) {
    // printf("%d ", Mmg[i]);
  // }
  // printf("\n");
  // //printf("\n");

  // srand(0);
  // int* tmp;
  // int tid = 0;
  // for (int i = 0; i < nnz; i++) {
    // //fscanf(fp, "%d %d %d", &M[0][i], &M[1][i], &M[2][i]);
    // //fread(tmp, sizeof(int), 3, fp);
    // tmp = data_dump + i*3;
 
    // M[0][i] = tmp[0] - 1;
    // M[1][i] = tmp[1] - 1;
    // M[2][i] = tmp[2];
    // //printf("%d %d %d \n", M[0][i], M[1][i], M[2][i]);
    // //M[2][i] = tmp[2] + (int)( (float)rand()/(float)RAND_MAX*32);

    // if (i > 0) {
      // if (tmp[0] < M[0][i-1]) { //not sorted order
        // printf("Found edge %d - %d after %d - %d : position in file %d\n", tmp[0], tmp[1], M[0][i-1], M[1][i-1], i);
        // printf("Input edge list not sorted according to starting vertex. Quitting now. \n");
        // exit(1);
      // }
    // }

    // while (M[0][i] >= Mmg[tid]) {
      // Mg[tid] = i;
      // printf("Mg[%d] = %d\n", tid, Mg[tid]);
      // tid++;
    // }
  // }
  // if (nnz <= 20) for (int i = 0; i < nnz; i++) printf("%d %d %d \n", M[0][i], M[1][i], M[2][i]);

  // Mg[tid] = nnz;
  // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
  // for (tid = tid + 1; tid <= g; tid++) {
    // Mg[tid] = nnz;
    // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
  // }
  // //printf("--------\n");
  // fclose(fp);

  // _mm_free(data_dump);

  // return n;
// }

// int ReadAndTranspose(char *fname, int*** _M, int **_Mg, int** _Mmg ,int* _g, int* m1) {
  // FILE* fp = fopen(fname, "r");
  // int m, n, nnz;
  // fscanf(fp, "%d %d %d", &m, &n, &nnz);
  // *m1 = n;

  // if (*_g > n) {
    // *_g = n;
  // }
  // int g = *_g;

  // *_M = (int**) _mm_malloc(sizeof(int*) * 3, 64);
  // *_Mg = (int*) _mm_malloc(sizeof(int) * (g + 1), 64);
  // *_Mmg = (int*) _mm_malloc(sizeof(int) * (g + 1), 64);

  // int** M = *_M;
  // int *Mmg = *_Mmg;
  // int *Mg = *_Mg;

  // M[0] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
  // M[1] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
  // M[2] = (int*) _mm_malloc(sizeof(int) * nnz, 64);

  // int r = n % g;
  // int q = n / g;
  // Mmg[0] = 1;
  // //printf("%d ", Mmg[0]);
  // for (int i = 1; i <= g; i++) {
    // if (r) {
      // Mmg[i] = Mmg[i - 1] + q + 1;
      // //printf("%d ", Mmg[i]);
      // r--;
    // } else {
      // Mmg[i] = Mmg[i - 1] + q;
      // //printf("%d ", Mmg[i]);
    // }
  // }
  // //printf("\n");
  // /*
  // int **A = (int**) _mm_malloc(sizeof(int*) * 3, 64);
  // A[0] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
  // A[1] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
  // A[2] = (int*) _mm_malloc(sizeof(int) * nnz, 64);

  // int *t = (int*) _mm_malloc(sizeof(int) * (n + 1), 64);
  // */
  // int* A[3];
  // A[0] = new int[nnz];
  // A[1] = new int[nnz];
  // A[2] = new int[nnz];
  // int* t = new int[n + 1];

  // for (int i = 0; i <= n; i++) {
    // t[i] = 0;
  // }

  // for (int i = 0; i < nnz; i++) {
    // fscanf(fp, "%d %d %d", &A[0][i], &A[1][i], &A[2][i]);
    // t[A[1][i]]++;
  // }

  // //int *j = (int*) _mm_malloc(sizeof(int) * (n + 1), 64);
  // int* j = new int[n + 1];

  // j[0] = 0;
  // for (int i = 1; i <= n; i++) {
    // j[i] = 0;
    // t[i] += t[i-1];
  // }

  // for (int i = 0; i < nnz; i++) {
    // int x = t[A[1][i]-1] + j[A[1][i]];
    // M[0][x] = A[1][i];
    // M[1][x] = A[0][i];
    // M[2][x] = A[2][i];
    // j[A[1][i]]++;
  // }
  // delete A[0];
  // delete A[1];
  // delete A[2];
  // delete t;
  // delete j;

  // int tid = 0;
  // for (int i = 0; i < nnz; i++) {
    // while (M[0][i] >= Mmg[tid]) {
      // Mg[tid] = i;
      // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
      // tid++;
    // }
  // }
  // Mg[tid] = nnz;
  // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
  // for (tid = tid + 1; tid <= g; tid++) {
    // Mg[tid] = nnz;
    // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
  // }
  // //printf("--------\n");
  // fclose(fp);

  // return m;
// }

// int ReadAndTransposeBinary(char *fname, int*** _M, int **_Mg, int** _Mmg ,int* _g, int* m1) {
  // FILE* fp = fopen(fname, "rb");
  // int m, n, nnz;
  // //fscanf(fp, "%d %d %d", &m, &n, &nnz);
  // int tmp_[3];
  // fread(tmp_, sizeof(int), 3, fp);
  // m = tmp_[0];
  // n = tmp_[1];
  // nnz = tmp_[2];

  // *m1 = n;

  // if (*_g > n) {
    // *_g = n;
  // }
  // int g = *_g;

  // *_M = (int**) _mm_malloc(sizeof(int*) * 3, 64);
  // *_Mg = (int*) _mm_malloc(sizeof(int) * (g + 1), 64);
  // *_Mmg = (int*) _mm_malloc(sizeof(int) * (g + 1), 64);

  // int** M = *_M;
  // int *Mmg = *_Mmg;
  // int *Mg = *_Mg;

  // size_t nnz_l = nnz;
  // M[0] = (int*) _mm_malloc(sizeof(int) * nnz_l, 64);
  // M[1] = (int*) _mm_malloc(sizeof(int) * nnz_l, 64);
  // M[2] = (int*) _mm_malloc(sizeof(int) * nnz_l, 64);

  // int* data_dump = (int*) _mm_malloc(sizeof(int)*nnz_l*3, 64);
  // fread(data_dump, sizeof(int), nnz_l*3, fp);

  // int r = n % g;
  // int q = n / g;
  // //int q512 = ((q+511)/512)*512; //q rounded up to 512
  // //int r = n - q512*g;
  // int n512 = (n/512)/g;
  // if (n512 == 0) n512 = 1;

  // Mmg[0] = 0;
  // for (int i = 1; i < g; i++) {
    // Mmg[i] = std::min(n512*i*512, n);
    // //printf("%d ", Mmg[i]);
  // }
  // Mmg[g] = n;
  
  // for (int i = 0; i <= g; i++) {
    // printf("%d ", Mmg[i]);
  // }
  // printf("\n");
  // /*Mmg[0] = 1;
  // //printf("%d ", Mmg[0]);
  // for (int i = 1; i <= g; i++) {
    // if (r) {
      // Mmg[i] = Mmg[i - 1] + q + 1;
      // //printf("%d ", Mmg[i]);
      // r--;
    // } else {
      // Mmg[i] = Mmg[i - 1] + q;
    // }
    // printf("%d ", Mmg[i]);
  // }*/
  // //printf("\n");
  // /*
  // int **A = (int**) _mm_malloc(sizeof(int*) * 3, 64);
  // A[0] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
  // A[1] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
  // A[2] = (int*) _mm_malloc(sizeof(int) * nnz, 64);

  // int *t = (int*) _mm_malloc(sizeof(int) * (n + 1), 64);
  // */
  // int* tmp;
  // int* A[3];
  // A[0] = new int[nnz];
  // A[1] = new int[nnz];
  // A[2] = new int[nnz];
  // int* t = new int[n+1];

  // for (int i = 0; i <= n; i++) {
    // t[i] = 0;
  // }

  // srand(0);
  // for (int i = 0; i < nnz; i++) {
    // //fscanf(fp, "%d %d %d", &A[0][i], &A[1][i], &A[2][i]);
    // //fread(tmp, sizeof(int), 3, fp);
    // tmp = data_dump + i*3;

    // A[0][i] = tmp[0] - 1;
    // A[1][i] = tmp[1] - 1;
    // A[2][i] = tmp[2];
    // //A[2][i] = tmp[2] + (int)( (float)rand()/(float)RAND_MAX*32);

    // t[A[1][i]+1]++;
  // }

  // //int *j = (int*) _mm_malloc(sizeof(int) * (n + 1), 64);
  // int* j = new int[n + 1];

  // j[0] = 0;
  // for (int i = 1; i < n; i++) {
    // j[i] = 0;
    // t[i] += t[i-1];
  // }

  // for (int i = 0; i < nnz; i++) {
    // int x = t[A[1][i]] + j[A[1][i]+1];
    // M[0][x] = A[1][i];
    // M[1][x] = A[0][i];
    // M[2][x] = A[2][i];
    // j[A[1][i]+1]++;
  // }
  // delete [] A[0];
  // delete [] A[1];
  // delete [] A[2];
  // delete [] t;
  // delete [] j;

  // int tid = 0;
  // for (int i = 0; i < nnz; i++) {
    // while (M[0][i] >= Mmg[tid] && tid<=g) {
      // Mg[tid] = i;
      // printf("Mg[%d] = %d\n", tid, Mg[tid]);
      // tid++;
    // }
  // }
  // Mg[tid] = nnz;
  // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
  // for (tid = tid + 1; tid <= g; tid++) {
    // Mg[tid] = nnz;
    // //printf("Mg[%d] = %d\n", tid, Mg[tid]);
  // }
  // //printf("--------\n");
  // fclose(fp);
  // _mm_free(data_dump);

  // return m;
// }

template <typename E=int>
void print_edges(edge_t<E> * edges, int nedges)
{
  for(int edge_id = 0 ; edge_id < nedges ; edge_id++)
  {
    std::cout << edges[edge_id].src << ", " << edges[edge_id].dst << ", " << edges[edge_id].val << ", " << edges[edge_id].partition_id << std::endl;
  }
}

template <typename E=int>
void write_edges_binary(edge_t<E> * edges, char * fname, int m, int n, int nnz)
{
  std::ofstream outfile(fname, std::ios::out | std::ios::binary);
  if(outfile.is_open())
  {
    outfile.write((char*) &m, sizeof(int));
    outfile.write((char*) &n, sizeof(int));
    outfile.write((char*) &nnz, sizeof(int));
    for(int edge_id = 0 ; edge_id < nnz ; edge_id++)
    {
      outfile.write((char*) &(edges[edge_id].src), sizeof(int));
      outfile.write((char*) &(edges[edge_id].dst), sizeof(int));
      outfile.write((char*) &(edges[edge_id].val), sizeof(E));
    }
  }
  else
  {
    std::cout << "Could not open binary output file" << std::endl;
    exit(0);
  }
  outfile.close();
}

template <typename E=int>
void dcsc_to_edges(edge_t<E> ** &vals, int ** &row_inds, int ** &col_ptrs, int ** col_indices, 
                int * &nnzs, int * ncols,
                int num_partitions, edge_t<E> * edges, int * row_pointers, 
		int * edge_pointers, int n)
{
  int edge_id = 0;
  for(int p = 0 ; p < num_partitions ; p++)
  {
    E * val = vals[p];
    int * row_ind = row_inds[p];
    int * col_ptr = col_ptrs[p];
    int * col_index = col_indices[p];
    int nnz_partition = nnzs[p];
    int ncol_partition = ncols[p];
    std::cout << "ncol_partition: " << ncol_partition << std::endl;
    for(int j_index = 0 ; j_index < ncol_partition ; j_index++)
    {
      int j = col_index[j_index];
      // for each element in the column
      for(int i = col_ptr[j_index] ; i < col_ptr[j_index+1] ; i++)
      {
        edges[edge_id].src = row_pointers[p] + row_ind[i];
	edges[edge_id].dst = j+1;
	edges[edge_id].val = val[i];
	edges[edge_id].partition_id = p;
	edge_id++;
      }
    }
  }
}

template <typename E=int>
bool compare_notrans(const edge_t<E> & a, const edge_t<E> & b)
{
  if(a.partition_id < b.partition_id) return true;
  else if(a.partition_id > b.partition_id) return false;

  if(a.dst < b.dst) return true;
  else if(a.dst > b.dst) return false;
  
  if(a.src < b.src) return true;
  else if(a.src > b.src) return false;
  return false;
}

template <typename E=int>
bool compare_trans(const edge_t<E> & a, const edge_t<E> & b)
{
  // sort by partition id, dst id, src id
  bool res = a.partition_id < b.partition_id;
  if(a.partition_id == b.partition_id)
  {
    res = a.src < b.src;
    if(a.src == b.src)
    {
      res = a.dst < b.dst;
    }
  }  
  return res;
}

template <typename E=int>
void read_from_binary(const char * fname, int &m, int &n, int &nnz, edge_t<E> * &edges)
{
  std::ifstream fin(fname, std::ios::binary);
  if(fin.is_open())
  {
    // Get header
    fin.read((char*)&m, sizeof(int));
    fin.read((char*)&n, sizeof(int));
    fin.read((char*)&nnz, sizeof(int));
    std::cout << "Got graph with m=" << m << "\tn=" << n << "\tnnz=" << nnz << std::endl;

    // Create edge list
    edge_io_t<E> * edge_blob = (edge_io_t<E>*) _mm_malloc((long int)nnz*sizeof(edge_io_t<E>), 64);
    fin.read((char*)edge_blob, (long int)nnz*sizeof(edge_io_t<E>));

    //printf("First line %d %d %d \n", edge_blob[0], edge_blob[1], edge_blob[2]);

    edges = (edge_t<E>*) _mm_malloc((long int)nnz * sizeof(edge_t<E>), 64);
    #pragma omp parallel for
    for(unsigned long int edge_id = 0 ; edge_id < nnz ; edge_id++)
    {
      edges[edge_id].src = edge_blob[edge_id].src - 1; //move to 0-based
      edges[edge_id].dst = edge_blob[edge_id].dst - 1; //move to 0-based
      edges[edge_id].val = edge_blob[edge_id].val;
    }
    _mm_free(edge_blob);
  }
  else
  {
    std::cout << "Could not open file " << fname << std::endl;
    exit(0);
  }
  fin.close();
}

template <typename E=int>
void read_from_txt(char * fname, int &m, int &n, int &nnz, edge_t<E> * &edges)
{
  std::ifstream fin(fname);
  if(fin.is_open())
  {
    std::string ln;

    // Get header
    getline(fin, ln);
    std::istringstream ln_ss(ln);
    ln_ss >> m;
    ln_ss >> n;
    ln_ss >> nnz;
    std::cout << "Got graph with m=" << m << "\tn=" << n << "\tnnz=" << nnz << std::endl;

    // Create edge list
    edges = (edge_t<E>*) _mm_malloc((long int)nnz * sizeof(edge_t<E>), 64);
    int edge_id = 0;
    int max_src = 0;
    while(getline(fin, ln))
    {
      std::istringstream ln_ss(ln);
      ln_ss >> edges[edge_id].src;
      edges[edge_id].src--;
      ln_ss >> edges[edge_id].dst;
      edges[edge_id].dst--;
      ln_ss >> edges[edge_id].val;
      edge_id++;
    }
  }
  else
  {
    std::cout << "File did not open" << std::endl;
    exit(0);
  }
  fin.close();
}

void static_partition(int * &row_pointers, int m, int num_partitions, int round)
{
    row_pointers = new int[num_partitions+1];

    if(round == 1)
    {
      int rows_per_partition = m / num_partitions;
      int rows_leftover = m % num_partitions;
      row_pointers[0] = 0;
      int current_row = row_pointers[0] + rows_per_partition;
      for(int p = 1 ; p < num_partitions+1 ; p++)
      {
        if(rows_leftover > 0)
        {
  	  current_row += 1;
          row_pointers[p] = current_row;
          current_row += rows_per_partition;
	  rows_leftover--;
        }
	else
	{
          row_pointers[p] = current_row;
          current_row += rows_per_partition;
	}
      }
    }
    else
    {
      int n512 = std::max((m / round) / num_partitions, 1);
      int n_round = std::max(0, m/round - n512*num_partitions);
      //printf("n_round = %d\n", n_round);
      assert(n_round < num_partitions);
      row_pointers[0] = 0;
      for(int p = 1 ; p < num_partitions ; p++)
      {
        //row_pointers[p] = std::min(n512*p*round,m);
        row_pointers[p] = row_pointers[p-1] + ((n_round>0)?((n512 + 1)*round):(n512*round));
        row_pointers[p] = std::min(row_pointers[p], m);
        if (n_round > 0) n_round--;
      }
      row_pointers[num_partitions] = m;
    }
  /*for (int i = 0; i <= num_partitions; i++) {
    printf("%d ", row_pointers[i]);
  }
  printf("\n");*/
}

template <typename E=int>
void set_edge_pointers(edge_t<E> * edges, int * row_pointers, int * &edge_pointers, 
                       int nnz, int num_partitions)
{
  // Figure out edge pointers
  edge_pointers = new int[num_partitions+1];
#define BINARY_SEARCH_EDGE_POINTERS
#ifndef BINARY_SEARCH_EDGE_POINTERS
  int p = 0;
  for(int edge_id = 0 ; edge_id < nnz ; edge_id++)
  {
    while(edges[edge_id].src >= row_pointers[p])
    {
      edge_pointers[p] = edge_id;
      p++;
    }
  }
  edge_pointers[p] = nnz;
  for(p = p+1 ; p < num_partitions+1 ; p++)
  {
    edge_pointers[p] = nnz;
  }
#else
  //#pragma omp parallel for
  for(int p = 0 ; p < num_partitions ; p++)
  {
    // binary search
    int e1 = 0;
    int e2 = nnz;
    int eh;
    while(e2 >= e1)
    {
      eh = e2 - (e2 - e1) / 2;
      if(eh == 0) 
      {  
        break;
      }
      if((edges[eh-1].src < row_pointers[p]) && edges[eh].src >= row_pointers[p])
      {
        break;
      }
      else if(edges[eh].src >= row_pointers[p])
      {
        e2 = eh-1;
      }
      else
      {
        e1 = eh+1;
      }
    }
    edge_pointers[p] = eh;
    //std::cout << edge_pointers[p] << "\t" << eh << std::endl;
  }
  edge_pointers[num_partitions] = nnz;
//#define CHECK_EDGE_POINTERS
#ifdef CHECK_EDGE_POINTERS
  int p = 0;
  for(int edge_id = 0 ; edge_id < nnz ; edge_id++)
  {
    while(edges[edge_id].src >= row_pointers[p])
    {
      assert(edge_pointers[p] == edge_id);
      p++;
    }
  }
  assert(edge_pointers[p] == nnz);
  for(p = p+1 ; p < num_partitions+1 ; p++)
  {
    assert(edge_pointers[p] == nnz);
  }
#endif // CHECK_EDGE_POINTERS
#endif
}

template <typename E=int>
void build_dcsc(E ** &vals, int ** &row_inds, int ** &col_ptrs, int ** &col_indices,
                int * &nnzs, int * &ncols,
                int num_partitions, edge_t<E> * edges, int * row_pointers, 
		int * edge_pointers, int n)
{
  vals = new E*[num_partitions];
  row_inds = new int*[num_partitions];
  col_ptrs = new int*[num_partitions];
  col_indices = new int*[num_partitions];
  nnzs = new int[num_partitions];
  ncols = new int[num_partitions];

  #pragma omp parallel for
  for(int p = 0 ; p < num_partitions ; p++)
  {
    int nnz_partition = nnzs[p] = edge_pointers[p+1] - edge_pointers[p];
    vals[p] = (E*) _mm_malloc(nnz_partition * sizeof(E), 64);
    row_inds[p] = (int*) _mm_malloc(nnz_partition * sizeof(int), 64);
    E * val = vals[p];
    int * row_ind = row_inds[p];

    int current_column = -1;
    int num_columns = 0;
    for(int edge_id = edge_pointers[p] ; edge_id < edge_pointers[p+1] ; edge_id++)
    {
      if(current_column < edges[edge_id].dst)
      {
        num_columns++;
	current_column = edges[edge_id].dst;
      }
    }
    ncols[p] = num_columns;
    col_indices[p] = (int*) _mm_malloc((num_columns+1) * sizeof(int), 64);
    col_ptrs[p] = (int*) _mm_malloc((num_columns+1) * sizeof(int), 64);
    int * col_index = col_indices[p];
    int * col_ptr = col_ptrs[p];
    current_column = -1;
    int current_column_num = -1;
    for(int edge_id = edge_pointers[p] ; edge_id < edge_pointers[p+1] ; edge_id++)
    {
      val[edge_id - edge_pointers[p]] = edges[edge_id].val;
      row_ind[edge_id - edge_pointers[p]] = edges[edge_id].src; 
      if(current_column < edges[edge_id].dst)
      {
        current_column_num++;
	current_column = edges[edge_id].dst;
	col_index[current_column_num] = current_column;
	col_ptr[current_column_num] = edge_id - edge_pointers[p];
      }
    }
    col_ptr[num_columns] = nnz_partition;
    col_index[num_columns] = n+1;
  }
}

template <typename E=int>
void partition_and_build_dcsc(int * &row_pointers,
                              int * &edge_pointers,
  			      E ** &vals,
			      int ** &row_inds,
			      int ** &col_ptrs,
			      int ** &col_indices,
			      int * &nnzs, 
			      int * &ncols, 
			      edge_t<E> * edges,
			      int m, 
			      int n,
			      int num_partitions,
			      int nnz,
			      int round)
{
  static_partition(row_pointers, m, num_partitions, round);

  struct timeval start, end;

  gettimeofday(&start, NULL);
  // Set partition ids
  #pragma omp parallel for
  for(int edge_id = 0 ; edge_id < nnz ; edge_id++)
  {
#define SET_PARTITION_IDS_BINARY_SEARCH
#ifndef SET_PARTITION_IDS_BINARY_SEARCH
    for(int p = 0 ; p < num_partitions ; p++)
    {
      if(edges[edge_id].src >= row_pointers[p] && edges[edge_id].src < row_pointers[p+1])
      {
        edges[edge_id].partition_id = p;
      }
    }
#else
    int key = edges[edge_id].src;
    int min_p = 0;
    int max_p = num_partitions-1;
    int h_p;
    while(max_p >= min_p)
    {
      h_p = max_p - ((max_p - min_p) / 2);
      if(key >= row_pointers[h_p] && key < row_pointers[h_p+1]) 
      {
        break;
      }
      else if(key >= row_pointers[h_p])
      {
        min_p = h_p+1;
      }
      else
      {
        max_p = h_p-1;
      }
    }
    edges[edge_id].partition_id = h_p;
//#define CHECK_PARTITION_IDS
#ifdef CHECK_PARTITION_IDS
    for(int p = 0 ; p < num_partitions ; p++)
    {
      if(edges[edge_id].src >= row_pointers[p] && edges[edge_id].src < row_pointers[p+1])
      {
        assert(edges[edge_id].partition_id == p);
      }
    }
#endif // CHECK_PARTITION_IDS
#endif
  }
  gettimeofday(&end, NULL);
  std::cout << "Finished setting ids, time: " << sec(start,end)  << std::endl;

  unsigned long int nnz_l = nnz;

  // Sort edge list
  std::cout << "Starting sort" << std::endl;
  gettimeofday(&start, NULL);
  __gnu_parallel::sort(edges, edges+nnz_l, compare_notrans<E>);
  gettimeofday(&end, NULL);
  std::cout << "Finished sort, time: " << sec(start,end)  << std::endl;

  //std::cout << "Sorted graph begin" << std::endl;
  //print_edges(edges, 9);
  //std::cout << "Sorted graph end" << std::endl;

  gettimeofday(&start, NULL);
  set_edge_pointers<E>(edges, row_pointers, edge_pointers, nnz, num_partitions);
  gettimeofday(&end, NULL);
  std::cout << "Finished setting edge pointers, time: " << sec(start,end)  << std::endl;

  for(int p = 0 ; p < num_partitions+1 ; p++)
  {
    //std::cout << "p: " << p << "\t edge pointer: " << edge_pointers[p] << std::endl;
  }

  // build DCSC
  std::cout << "Starting build_dcsc" << std::endl;
  gettimeofday(&start, NULL);
  build_dcsc<E>(vals, row_inds, col_ptrs, col_indices, nnzs, ncols, num_partitions, edges, row_pointers, edge_pointers, n);
  gettimeofday(&end, NULL);
  std::cout << "Finished build_dcsc, time: " << sec(start,end)  << std::endl;


}

#define CHECK(a,b,c) { if((a)!=(b)) {std::cout << a << " " << b << " ERROR: " << c << std::endl; exit(0);} }

template<class V, class E>
void Graph<V,E>::ReadMTX(const char* filename, int grid_size) {

//#define USE_SORTED_INPUT
// #ifdef USE_SORTED_INPUT
  ReadMTX_sort(filename, grid_size); 
// #else
  // ReadMTX_old(filename, grid_size); 
// #endif
  
// //#define CHECK_SORTED_INPUT
// #ifdef CHECK_SORTED_INPUT
  // Graph<V> G_sort;
  // G_sort.ReadMTX_sort(filename, grid_size); 
  
  // // Check variables
  // int grid_m = grid_size;
  // if(grid_m > mat[0]->m) grid_m = mat[0]->m;
  // if(grid_m > mat[0]->n) grid_m = mat[0]->n;
  // for(int i = 0 ; i < grid_m ; i++)
  // {
    // printf("Partition mat %d\n", i);
    // CHECK(mat[i]->m, G_sort.mat[i]->m, "m");
    // CHECK(mat[i]->n, G_sort.mat[i]->n, "n");
    // CHECK(mat[i]->nnz, G_sort.mat[i]->nnz, "nnz");
    // CHECK(mat[i]->nzx, G_sort.mat[i]->nzx, "nzx");
    // CHECK(mat[i]->lock, G_sort.mat[i]->lock, "lock");
    // CHECK(mat[i]->isColumn, G_sort.mat[i]->isColumn, "isColumn");

    // // Check arrays
    // for(int j = 0 ; j < mat[i]->nnz ; j++)
    // {
      // CHECK(mat[i]->yindex[j], G_sort.mat[i]->yindex[j], "yindex");
      // CHECK(mat[i]->value[j], G_sort.mat[i]->value[j], "value");
    // }

    // for(int j = 0 ; j < mat[i]->nzx ; j++)
    // {
      // CHECK(mat[i]->xindex[j], G_sort.mat[i]->xindex[j], "xindex");
      // CHECK(mat[i]->starty[j], G_sort.mat[i]->starty[j], "starty");
    // }
  // }

  // for(int i = 0 ; i < grid_m ; i++)
  // {
    // printf("Partition matT %d\n", i);
    // CHECK(matT[i]->m, G_sort.matT[i]->m, "m");
    // CHECK(matT[i]->n, G_sort.matT[i]->n, "n");
    // CHECK(matT[i]->nnz, G_sort.matT[i]->nnz, "nnz");
    // CHECK(matT[i]->nzx, G_sort.matT[i]->nzx, "nzx");
    // CHECK(matT[i]->lock, G_sort.matT[i]->lock, "lock");
    // CHECK(matT[i]->isColumn, G_sort.matT[i]->isColumn, "isColumn");

    // // Check arrays
    // for(int j = 0 ; j < matT[i]->nnz ; j++)
    // {
      // CHECK(matT[i]->yindex[j], G_sort.matT[i]->yindex[j], "yindex");
      // CHECK(matT[i]->value[j], G_sort.matT[i]->value[j], "value");
    // }

    // for(int j = 0 ; j < matT[i]->nzx ; j++)
    // {
      // CHECK(matT[i]->xindex[j], G_sort.matT[i]->xindex[j], "xindex");
      // CHECK(matT[i]->starty[j], G_sort.matT[i]->starty[j], "starty");
    // }
  // }

  // // Check graph
  // CHECK(nvertices, G_sort.nvertices, "nvertices");
  // CHECK(nnz, G_sort.nnz, "nnz");
  // CHECK(vertexpropertyowner, G_sort.vertexpropertyowner, "vertexpropertyowner");
  // for(int j = 0 ; j < mat[0]->m ; j++)
  // {
    // CHECK(active[j], G_sort.active[j], "active");
    // CHECK(id[j], G_sort.id[j], "id");
  // }
  // std::cout << "Test passed" << std::endl;
// #endif

  return;
}


template<class V, class E>
void Graph<V,E>::ReadMTX_sort(const char* filename, int grid_size) {

  struct timeval start, end;
  gettimeofday(&start, 0);

  int m_, n_, nnz_;

  // Insert my code here
  edge_t<E> * edges;
  std::cout << "Starting file read of " << filename << std::endl;
  gettimeofday(&start, NULL);
  read_from_binary<E>(filename, m_, n_, nnz_, edges);
  gettimeofday(&end, NULL);
  std::cout << "Finished file read of " << filename << ", time: " << sec(start,end)  << std::endl;

  ReadMTX_sort(edges, m_, n_, nnz_, grid_size);

  _mm_free((void*)edges);

  gettimeofday(&end, 0);
  double time = (end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)*1e-6;
  printf("Completed reading A from file in %lf seconds.\n", time);
}

template<class V, class E>
void Graph<V,E>::ReadMTX_sort(edge_t<E>* edges, int m_, int n_, int nnz_, int grid_size, int alloc) {

  struct timeval start, end;
  gettimeofday(&start, 0);

  int * row_pointers_notrans;
  int * edge_pointers_notrans;
  E ** vals_notrans;
  int ** row_inds_notrans;
  int ** col_ptrs_notrans;
  int ** col_indices_notrans;
  int * ncols_notrans;
  int * nnzs_notrans;

  int * row_pointers_trans;
  int * edge_pointers_trans;
  E ** vals_trans;
  int ** row_inds_trans;
  int ** col_ptrs_trans;
  int ** col_indices_trans;
  int * ncols_trans;
  int * nnzs_trans;
  //int m_, n_, nnz_;


  // Insert my code here
/*
  edge_t * edges;
  std::cout << "Starting file read" << std::endl;
  gettimeofday(&start, NULL);
  read_from_binary(filename, m_, n_, nnz_, edges);
  gettimeofday(&end, NULL);
  std::cout << "Finished file read, time: " << sec(start,end)  << std::endl;
  */
  if(grid_size > m_) grid_size = m_;
  if(grid_size > n_) grid_size = n_;

  int round = 512;
  
  grid_size = std::min((m_+round-1)/round, grid_size);
  grid_size = std::min((n_+round-1)/round, grid_size);
  

  partition_and_build_dcsc<E>(row_pointers_notrans,
                           edge_pointers_notrans,
			   vals_notrans,
			   row_inds_notrans,
			   col_ptrs_notrans,
			   col_indices_notrans,
			   nnzs_notrans,
			   ncols_notrans,
			   edges,
			   m_,
			   n_,
			   grid_size,
			   nnz_,
			   round);
			   //1);

  #pragma omp parallel for
  for(int edge_id = 0 ; edge_id < nnz_ ; edge_id++)
  {
    int tmp = edges[edge_id].src;
    edges[edge_id].src = edges[edge_id].dst;
    edges[edge_id].dst = tmp;
  }

  partition_and_build_dcsc<E>(row_pointers_trans,
                           edge_pointers_trans,
			   vals_trans,
			   row_inds_trans,
			   col_ptrs_trans,
			   col_indices_trans,
			   nnzs_trans,
			   ncols_trans,
			   edges,
			   n_,
			   m_,
			   grid_size,
			   nnz_,
			   round);

  nnz = 0;
  mat = new MatrixDC<E>*[grid_size];
  matT = new MatrixDC<E>*[grid_size];
  start_src_vertices = new int[grid_size];
  end_src_vertices = new int[grid_size];
  nparts = grid_size;
 
  #pragma omp parallel for
  for (int p = 0; p < grid_size; p++) 
  {
    int n_notrans = row_pointers_notrans[p+1] - row_pointers_notrans[p];
    int n_trans = row_pointers_trans[p+1] - row_pointers_trans[p];
    mat[p] = new MatrixDC<E>(m_, n_notrans, nnzs_notrans[p], false, 
                            ncols_notrans[p], col_indices_notrans[p],
			    col_ptrs_notrans[p], row_inds_notrans[p], 
			    vals_notrans[p]);
    matT[p] = new MatrixDC<E>(n_, n_trans, nnzs_trans[p], false, 
                            ncols_trans[p], col_indices_trans[p],
			    col_ptrs_trans[p], row_inds_trans[p], 
			    vals_trans[p]);
    start_src_vertices[p] = row_pointers_trans[p];
    end_src_vertices[p] = row_pointers_trans[p+1]-1;
  }

  delete [] ncols_notrans;
  delete [] col_indices_notrans;
  delete [] col_ptrs_notrans;
  delete [] row_inds_notrans;
  delete [] vals_notrans;

  delete [] ncols_trans;
  delete [] col_indices_trans;
  delete [] col_ptrs_trans;
  delete [] row_inds_trans;
  delete [] vals_trans;

  for (int i = 0; i < grid_size; i++) {
    nnz += mat[i]->nnz;
  }

  if(alloc) {
    vertexproperty = new V[m_](); 
    active = new bool[m_];
    //id = new unsigned long long int[m_];

    for (int i = 0; i < m_; i++) {
      active[i] = false;
      //id[i] = i;
    }
  } else {
    vertexproperty = NULL; 
    active = NULL;
    //id = NULL;
  }

  nvertices = m_;
  vertexpropertyowner = 1;

  gettimeofday(&end, 0);
  double time = (end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)*1e-6;
  printf("Completed reading A from memory in %lf seconds.\n", time);
}



// template<class V, class E>
// void Graph<V,E>::ReadMTX_old(const char* filename, int grid_size) {

  // //if (grid_size > MAX_PARTS) grid_size = MAX_PARTS;

  // nparts = grid_size;
  // int grid_m = grid_size; 
  // //int nthreads = nparts;

  // struct timeval start, end;
  // //unsigned long long int start = _rdtsc();
  // gettimeofday(&start, 0);
  // int **A = NULL;
  // int *Amg = NULL;
  // int *Ag = NULL;
  // int **B = NULL;
  // int *Bmg = NULL;
  // int *Bg = NULL;

  // int am;
  // int an;
  // int bm, bn;
  // //an = Read(filename, &A, &Ag, &Amg, &grid_m, &am);
  // //bn = ReadAndTranspose(filename, &B, &Bg, &Bmg, &grid_m, &bm);
  // an = ReadBinary(filename, &A, &Ag, &Amg, &grid_m, &am);
  // bn = ReadAndTransposeBinary(filename, &B, &Bg, &Bmg, &grid_m, &bm);

  // //unsigned long long int now = _rdtsc();
  // gettimeofday(&end, 0);
  // //printf("Completed reading A in %lf seconds on Xeon.\n", (now - start) / 2.7 / pow(10, 9));
  // double time = (end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)*1e-6;
  // printf("Completed reading A in %lf seconds.\n", time);
  // nnz = 0;

  // //now = _rdtsc();
  // gettimeofday(&start, 0);

  // //MatrixDC* A_DCSC[grid_m];
  // mat = new MatrixDC<E>*[grid_m];
  // matT = new MatrixDC<E>*[grid_m];
  // start_src_vertices = new int[grid_m];
  // end_src_vertices = new int[grid_m];
 
  // //#pragma omp parallel for num_threads(nthreads)
  // #pragma omp parallel for //num_threads(8)
  // for (int aid = 0; aid < grid_m; aid++) 
  // {
    // //int tid = omp_get_thread_num();
      // //unsigned long long int begin = _rdtsc();
      // //if (tid < grid_m) {
	// //int aid = tid;
        // //printf("amg aid %d %d\n", Amg[aid], Amg[aid + 1]);
        // mat[aid] = MatrixDC<E>::ReadDCSXRev(A, Ag, Amg[aid + 1] - Amg[aid], an, aid);
        // matT[aid] = MatrixDC<E>::ReadDCSXRev(B, Bg, Bmg[aid + 1] - Bmg[aid], bn, aid);
      // //} 
      // start_src_vertices[aid] = Bmg[aid];
      // end_src_vertices[aid] = Bmg[aid+1]-1;
  // }


  // for (int i = 0; i < grid_m; i++) {
    // nnz += mat[i]->nnz;
    
    // //printf("NNZ[%d] :: A %d At %d \n", i, mat[i]->nnz, matT[i]->nnz);
    // //mat[i]->printStats();
    // //matT[i]->printStats();
    // //printf("Part %d Start %d end %d \n", i, start_src_vertices[i], start_src_vertices[i+1]-1);
  // }

  // //unsigned long long int now1 = _rdtsc();
  // gettimeofday(&end, 0);
  // time = (end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)*1e-6;

  // //printf("Reading into %d DCSC blocks completed in %lf seconds on Xeon.\n", grid_m, (now1 - now) / 2.7 / pow(10, 9));
  // printf("Reading into %d DCSC blocks completed in %lf seconds.\n", grid_m, time);
  // printf("Matrix size: %d x %d\n", am, an);
  // printf("Matrix size: %d x %d\n", bm, bn);

  // _mm_free(A[0]);
  // _mm_free(A[1]);
  // _mm_free(A[2]);
  // _mm_free(A);
  // _mm_free(Ag);
  // _mm_free(Amg);
  // _mm_free(B[0]);
  // _mm_free(B[1]);
  // _mm_free(B[2]);
  // _mm_free(B);
  // _mm_free(Bg);
  // _mm_free(Bmg);
  
  // vertexproperty = new V[am](); 
  // active = new bool[am];
  // id = new unsigned long long int[am];

  // for (int i = 0; i < am; i++) {
    // active[i] = false;
    // id[i] = i;
  // }
  // nvertices = am;
  // vertexpropertyowner = 1;
// }

template<class V, class E> 
void Graph<V,E>::setAllActive() {
  //for (int i = 0; i <= nvertices; i++) {
  //  active[i] = true;
  //}
  memset(active, 0xff, sizeof(bool)*(nvertices));
}

template<class V, class E> 
void Graph<V,E>::setAllInactive() {
  memset(active, 0x0, sizeof(bool)*(nvertices));
}

template<class V, class E> 
void Graph<V,E>::setActive(int v) {
  active[v-1] = true;
}

template<class V, class E> 
void Graph<V,E>::setInactive(int v) {
  active[v-1] = false;
}
template<class V, class E> 
void Graph<V,E>::reset() {
  memset(active, 0, sizeof(bool)*(nvertices));

  #pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < nvertices; i++) {
    V v;
    vertexproperty[i] = v;
  }
}

template<class V, class E> 
void Graph<V,E>::shareVertexProperty(Graph<V,E>& g) {
  delete [] vertexproperty;
  vertexproperty = g.vertexproperty;
  vertexpropertyowner = 0;
}

template<class V, class E> 
void Graph<V,E>::setAllVertexproperty(const V& val) {
  #pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < nvertices; i++) {
    vertexproperty[i] = val;
  }
}

template<class V, class E> 
void Graph<V,E>::setVertexproperty(int v, const V& val) {
  vertexproperty[v-1] = val;
}

template<class V, class E> 
V Graph<V,E>::getVertexproperty(int v) const {
  return vertexproperty[v-1];
}

int getId(const int i, const int* start, const int* end, const int n) {
  for (int j = 0; j < n; j++) {
    if (i >= start[j] && i <= end[j]) {
      return j;
    }
  }
}

template<class V, class E> 
int Graph<V,E>::getBlockIdBySrc(int vertexId) const {
  return getId(vertexId, start_src_vertices, end_src_vertices, nparts);
}

template<class V, class E> 
int Graph<V,E>::getBlockIdByDst(int vertexId) const {
  return getId(vertexId, start_dst_vertices, end_dst_vertices, nparts);
}

template<class V, class E> 
int Graph<V,E>::getNumberOfVertices() const {
  return nvertices;
}

template<class V, class E> 
void Graph<V,E>::applyToAllVertices(void (*ApplyFn)(V, V*, void*), void* param) {
  #pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < nvertices; i++) {
    ApplyFn(vertexproperty[i], &vertexproperty[i], param);
  }
}

template<class V, class E> 
template<class T> 
void Graph<V,E>::applyReduceAllVertices(T* val, void (*ApplyFn)(V*, T*, void*), void (*ReduceFn)(T,T,T*,void*), void* param) {
  T sum = *val;

  /*for (int i = 0; i < nvertices; i++) {
    T tmp;
    ApplyFn(vertexproperty[i], &tmp, param);
    ReduceFn(sum, tmp, &sum, param);
  }*/
  T* tmpsum = new T[nthreads*16]; //reduce false sharing
  for(int i = 0; i < nthreads; i++) tmpsum[i*16] = sum;
  #pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < nvertices; i++) {
    T tmp;
    int tid = omp_get_thread_num();
    ApplyFn(&vertexproperty[i], &tmp, param);
    ReduceFn(tmpsum[tid*16], tmp, &tmpsum[tid*16], param);
  }

  for(int i = 0; i < nthreads; i++) {
    ReduceFn(tmpsum[i*16], sum, &sum, param);
  }
  delete [] tmpsum;

  *val = sum;
}


template<class V, class E> 
Graph<V,E>::~Graph() {
  if (vertexpropertyowner) {
    if(vertexproperty) delete [] vertexproperty;
  }
  if (active) delete [] active;
  //if (id) delete [] id;
  if (start_src_vertices) delete [] start_src_vertices;
  if (end_src_vertices) delete [] end_src_vertices;

}

