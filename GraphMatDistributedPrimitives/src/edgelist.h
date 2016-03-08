/******************************************************************************
 * ** Copyright (c) 2016, Intel Corporation                                     **
 * ** All rights reserved.                                                      **
 * **                                                                           **
 * ** Redistribution and use in source and binary forms, with or without        **
 * ** modification, are permitted provided that the following conditions        **
 * ** are met:                                                                  **
 * ** 1. Redistributions of source code must retain the above copyright         **
 * **    notice, this list of conditions and the following disclaimer.          **
 * ** 2. Redistributions in binary form must reproduce the above copyright      **
 * **    notice, this list of conditions and the following disclaimer in the    **
 * **    documentation and/or other materials provided with the distribution.   **
 * ** 3. Neither the name of the copyright holder nor the names of its          **
 * **    contributors may be used to endorse or promote products derived        **
 * **    from this software without specific prior written permission.          **
 * **                                                                           **
 * ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
 * ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
 * ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
 * ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
 * ** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
 * ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
 * ** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
 * ** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
 * ** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
 * ** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
 * ** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * * ******************************************************************************/
/* Michael Anderson (Intel Corp.)
 *  * ******************************************************************************/


#ifndef SRC_EDGELIST_H_
#define SRC_EDGELIST_H_

template <typename T>
struct edge_t {
  edge_t() {}
  edge_t(int _src, int _dst, T _val)
  {
    src = _src;
    dst = _dst;
    val = _val;
  }
  int src;
  int dst;
  T val;
};

template <typename T>
struct edgelist_t {
  edge_t<T>* edges;
  int m;
  int n;
  int nnz;
  edgelist_t() : m(0), n(0), nnz(0) {}
  edgelist_t(int _m, int _n, int _nnz)
  {
    m = _m;
    n = _n;
    nnz = _nnz;
    edges = (edge_t<T>*)_mm_malloc(nnz * sizeof(edge_t<T>), 64);
  }
};

template <typename T>
struct tedge_t {
  int src;
  int dst;
  int tile_id;
  T val;
};

template <typename T>
void load_edgelist(const char* dir, int myrank, int nrank,
                   edgelist_t<T>* edgelist) {

  edgelist->nnz = 0;
  for(int i = global_myrank ; ; i += global_nrank)
  {
    std::stringstream fname_ss;
    fname_ss << dir << i;
    //printf("Opening file: %s\n", fname_ss.str().c_str());
    FILE* fp = fopen(fname_ss.str().c_str(), "r");
    if(!fp) break;

    int tmp_[3];
    fread(tmp_, sizeof(int), 3, fp);
    edgelist->m = tmp_[0];
    edgelist->n = tmp_[1];
    edgelist->nnz += tmp_[2];
    fclose(fp);
  }

  MPI_Bcast(&(edgelist->m), 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&(edgelist->n), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::cout << "Got: " << edgelist->nnz << " edges\n" << std::endl;
  
  edgelist->edges = reinterpret_cast<edge_t<T>*>(
      _mm_malloc((uint64_t)edgelist->nnz * (uint64_t)sizeof(edge_t<T>), 64));


  int nnzcnt = 0;
  for(int i = global_myrank ; ; i += global_nrank)
  {
    std::stringstream fname_ss;
    fname_ss << dir << i;
    //printf("Opening file: %s\n", fname_ss.str().c_str());
    FILE* fp = fopen(fname_ss.str().c_str(), "r");
    if(!fp) break;

    int tmp_[3];
    fread(tmp_, sizeof(int), 3, fp);
    assert(tmp_[0] == edgelist->m);
    assert(tmp_[1] == edgelist->n);

    fread(edgelist->edges + nnzcnt, sizeof(edge_t<T>), tmp_[2], fp);
    #ifdef __DEBUG
    for(int j = 0 ; j < tmp_[2] ; j++)
    {
      if(edgelist->edges[nnzcnt].src <= 0 ||
         edgelist->edges[nnzcnt].dst <= 0 ||
         edgelist->edges[nnzcnt].src > edgelist->m ||
         edgelist->edges[nnzcnt].dst > edgelist->n)
      {
        std::cout << "Invalid edge, i, j, nnz: " << i << " , " << j << " , " << nnzcnt << std::endl;
        exit(0);
      }
      nnzcnt++;
    }
    #else
    nnzcnt += tmp_[2];
    #endif

    fclose(fp);
  }
}

template <typename T>
void write_edgelist_txt(const char* dir, int myrank, int nrank, 
                       const edgelist_t<T> & edgelist)
{
  std::stringstream fname_ss;
  fname_ss << dir << global_myrank;
  printf("Opening file: %s\n", fname_ss.str().c_str());
  FILE * fp = fopen(fname_ss.str().c_str(), "w");
  fprintf(fp, "%d %d %u\n", edgelist.m, edgelist.n, edgelist.nnz);
  for(int i = 0 ; i < edgelist.nnz ; i++)
  {
    fprintf(fp, "%d %d %.15e\n", edgelist.edges[i].src, edgelist.edges[i].dst, edgelist.edges[i].val);
  }
  fclose(fp);
}

template <typename T>
void load_edgelist_txt(const char* dir, int myrank, int nrank,
                       edgelist_t<T>* edgelist) {
  edgelist->nnz = 0;
  for(int i = global_myrank ; ; i += global_nrank)
  {
    std::stringstream fname_ss;
    fname_ss << dir << i;
    //printf("Opening file: %s\n", fname_ss.str().c_str());
    FILE* fp = fopen(fname_ss.str().c_str(), "r");
    if(!fp) break;

    int tmp_[3];
    fscanf(fp, "%d %d %u", &(tmp_[0]), &(tmp_[1]), &(tmp_[2]));
    edgelist->m = tmp_[0];
    edgelist->n = tmp_[1];
    edgelist->nnz += tmp_[2];
    fclose(fp);
  }

  std::cout << "Got: " << edgelist->nnz << " edges\n" << std::endl;
  
  edgelist->edges = reinterpret_cast<edge_t<T>*>(
      _mm_malloc((uint64_t)edgelist->nnz * (uint64_t)sizeof(edge_t<T>), 64));


  int nnzcnt = 0;
  for(int i = global_myrank ; ; i += global_nrank)
  {
    std::stringstream fname_ss;
    fname_ss << dir << i;
    //printf("Opening file: %s\n", fname_ss.str().c_str());
    FILE* fp = fopen(fname_ss.str().c_str(), "r");
    if(!fp) break;

    int tmp_[3];
    fscanf(fp, "%d %d %u", &(tmp_[0]), &(tmp_[1]), &(tmp_[2]));
    assert(tmp_[0] == edgelist->m);
    assert(tmp_[1] == edgelist->n);
    for(int j = 0 ; j < tmp_[2] ; j++)
    {
      if(!fscanf(fp, "%d %d %lf", &(edgelist->edges[nnzcnt].src),
             &(edgelist->edges[nnzcnt].dst), &(edgelist->edges[nnzcnt].val)))
      {
        std::cout << "Bad edge read, i, j, nnzcnt " << i << " , " << j << " , " << nnzcnt << std::endl;
        exit(0);
      }

      if(edgelist->edges[nnzcnt].src <= 0 ||
         edgelist->edges[nnzcnt].dst <= 0 ||
         edgelist->edges[nnzcnt].src > edgelist->m ||
         edgelist->edges[nnzcnt].dst > edgelist->n)
      {
        std::cout << "Invalid edge, i, j, nnz: " << i << " , " << j << " , " << nnzcnt << std::endl;
        exit(0);
      }
      nnzcnt++;
    }
    fclose(fp);
  }
}

template <typename T>
void randomize_edgelist_square(edgelist_t<T>* edgelist, int nrank) {
  unsigned int* mapping = new unsigned int[edgelist->m];
  unsigned int* rval = new unsigned int[edgelist->m];

  if (global_myrank == 0) {
    srand(5);
    // #pragma omp parallel for
    for (int i = 0; i < edgelist->m; i++) {
      mapping[i] = i;
      rval[i] = rand() % edgelist->m;
    }

    for (int i = 0; i < edgelist->m; i++) {
      unsigned int tmp = mapping[i];
      mapping[i] = mapping[rval[i]];
      mapping[rval[i]] = tmp;
    }
  }
  delete[] rval;

  MPI_Bcast(mapping, edgelist->m, MPI_INT, 0, MPI_COMM_WORLD);

#pragma omp parallel for
  for (int i = 0; i < edgelist->nnz; i++) {
    edgelist->edges[i].src = mapping[edgelist->edges[i].src - 1] + 1;
    edgelist->edges[i].dst = mapping[edgelist->edges[i].dst - 1] + 1;
  }
  delete[] mapping;
}

template<typename T>
void remove_empty_columns(edgelist_t<T> * edges, int ** remaining_indices)
{
  // Remove empty columns
  bool * colexists = new bool[edges->n];
  memset(colexists, 0, edges->n * sizeof(bool));
  int * new_colids = new int[edges->n+1];
  memset(new_colids, 0, (edges->n + 1) * sizeof(int));
  int new_ncols = 0;
  for(int i = 0 ; i < edges->nnz ; i++)
  {
    if(!colexists[edges->edges[i].dst-1])
    {
      new_ncols++;
    }
    colexists[edges->edges[i].dst-1] = true;
  }
  std::cout << "New ncols: " << new_ncols << std::endl;
  *(remaining_indices) = (int*) _mm_malloc(new_ncols * sizeof(int), 64);
  int new_colcnt = 0;
  for(int i = 0 ; i < edges->n; i++)
  {
    new_colids[i+1] = (colexists[i] ? 1 : 0) + new_colids[i];
    if(colexists[i])
    {
      assert(new_colcnt < new_ncols);
      (*(remaining_indices))[new_colcnt] = i+1;
      new_colcnt++;
    }
  }
  assert(new_colcnt == new_ncols);
  #pragma omp parallel for
  for(int i = 0 ; i < edges->nnz ; i++)
  {
    edges->edges[i].dst = new_colids[edges->edges[i].dst-1] + 1;
    assert(edges->edges[i].dst - 1 >= 0);
    assert(edges->edges[i].dst - 1 < new_ncols);
  }
  edges->n = new_ncols;
  delete [] colexists;
  delete [] new_colids;
}

template<typename T>
void filter_edges_by_row(edgelist_t<T> * edges, int start_row, int end_row)
{
  int valid_edgecnt = 0;
  for(int i = 0 ; i < edges->nnz ; i++)
  {
    if(edges->edges[i].src-1 < end_row && 
       edges->edges[i].src-1 >= start_row)
    {
      edges->edges[valid_edgecnt] = edges->edges[i];
      edges->edges[valid_edgecnt].src -= start_row;
      valid_edgecnt++;
    }
  }
  edges->nnz = valid_edgecnt;
  edges->m = (end_row-start_row);
  std::cout << "New edges->m: " << edges->m << std::endl;
}

#endif  // SRC_EDGELIST_H_
