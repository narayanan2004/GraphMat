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
/* Michael Anderson (Intel Corp.), Narayanan Sundaram (Intel Corp.)
 *  * ******************************************************************************/


#ifndef SRC_EDGELIST_H_
#define SRC_EDGELIST_H_

#include <string>

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
  edgelist_t() : m(0), n(0), nnz(0), edges(nullptr) {}
  edgelist_t(int _m, int _n, int _nnz)
  {
    m = _m;
    n = _n;
    nnz = _nnz;
    if(nnz > 0) {
      edges = reinterpret_cast<edge_t<T>*>(_mm_malloc((size_t)nnz * sizeof(edge_t<T>), 64));
    }
  }
  edgelist_t(edge_t<T>* edges, int m, int n, int nnz) : edges(edges), m(m), n(n), nnz(nnz) {}
  void clear() {
    if (nnz > 0) {
      _mm_free(edges);
    }
    edges = nullptr;
    nnz = 0;
    m = 0;
    n = 0;
  }
};

template <typename T>
struct tedge_t {
  int src;
  int dst;
  int tile_id;
  T val;
};


template<typename T>
bool readLine (FILE * ifile, int * src, int * dst, T * val, bool binaryformat=true, bool edgeweights=true)
{
  if(binaryformat) {	
    auto fread_bytes = fread(src, sizeof(int), 1, ifile);
    if (feof(ifile)) return false;
    assert(fread_bytes == 1);
    fread_bytes = fread(dst, sizeof(int), 1, ifile);
    if (feof(ifile)) return false;
    assert(fread_bytes == 1);
    if (edgeweights) {
      fread_bytes = fread(val, sizeof(T), 1, ifile);
      if (feof(ifile)) return false;
      assert(fread_bytes == 1);
      *val = (T)(1);
    }
  } else {
    if (edgeweights) {
      int ret;
      if (std::is_same<T, float>::value) {
        ret = fscanf(ifile, "%d %d %f", src, dst, val);
        if (ret != 3) return false;
      } else if (std::is_same<T, double>::value) {
        ret = fscanf(ifile, "%d %d %lf", src, dst, val);
        if (ret != 3) return false;
      } else if (std::is_same<T, int>::value) {
        ret = fscanf(ifile, "%d %d %d", src, dst, val);
        if (ret != 3) return false;
      } else if (std::is_same<T, unsigned int>::value) {
        ret = fscanf(ifile, "%d %d %u", src, dst, val);
        if (ret != 3) return false;
      }else {
        std::cout << "Data type not supported (read)" << std::endl;
      }
    } else {
      int ret = fscanf(ifile, "%d %d", src, dst);
      if (ret == 2) {
        *val = (T)(1);
      } else return false;
    }
    if (feof(ifile)) return false;
  }
  return true;
}

template<typename T>
void get_maxid_and_nnz(FILE* fp, int* m, int* n, unsigned long int* nnz, bool binaryformat=true, bool header=true, bool edgeweights=true) {
  if (header) {
    int tmp_[3];
    if (binaryformat) {
      auto fread_bytes = fread(tmp_, sizeof(int), 3, fp);
      assert(fread_bytes == 3);
      *m = tmp_[0];
      *n = tmp_[1];
      *nnz = tmp_[2];
    } else {
      int ret = fscanf(fp, "%d %d %u", &(tmp_[0]), &(tmp_[1]), &(tmp_[2]));
      assert(ret == 3);
      *m = tmp_[0];
      *n = tmp_[1];
      *nnz = tmp_[2];
    }
    return;
  } else { //no header
    unsigned long nnz_ = 0;
    int tempsrc, tempdst;
    int maxm = 0;
    int maxn = 0;
    T tempval;
    while(true) {
      if(feof(fp)) {
        break;
      }
      if (!readLine<T>(fp, &tempsrc, &tempdst, &tempval, binaryformat, edgeweights)) {
        break;
      }
      maxm = (maxm > tempsrc)?(maxm):(tempsrc);
      maxn = (maxn > tempdst)?(maxn):(tempdst);
      nnz_++;
    }
    *m = maxm;
    *n = maxn;
    *nnz = nnz_;
  }
}


template<typename T>
void writeLine (FILE* ofile, int src, int dst, T val, bool binaryformat=true, bool edgeweights=true)
{
  if (binaryformat) {
      auto fwrite_bytes = fwrite(&src, sizeof(int), 1, ofile);
      assert(fwrite_bytes == 1);
      fwrite_bytes = fwrite(&dst, sizeof(int), 1, ofile);
      assert(fwrite_bytes == 1);
      if (edgeweights) {
        fwrite_bytes = fwrite(&val, sizeof(T), 1, ofile);
        assert(fwrite_bytes == 1);
      }
  } else {
    if (edgeweights) { 
      if (std::is_same<T, float>::value) {
        fprintf(ofile, "%d %d %.8f\n", src, dst, val);
      } else if (std::is_same<T, double>::value) {
        fprintf(ofile, "%d %d %.15lf\n", src, dst, val);
      } else if (std::is_same<T, int>::value) {
        fprintf(ofile, "%d %d %d\n", src, dst, val);
      } else if (std::is_same<T, unsigned int>::value) {
        fprintf(ofile, "%d %d %u\n", src, dst, val);
      } else {
        std::cout << "Data type not supported (write)\n";
      }
    } else {
      fprintf(ofile, "%d %d\n", src, dst);
    }
  }
}

template <typename T>
void write_edgelist(const char* dir, const edgelist_t<T> & edgelist,
                    bool binaryformat=true, bool header=true, bool edgeweights=true)
{
  int global_nrank = get_global_nrank();
  int global_myrank = get_global_myrank();
  std::stringstream fname_ss;
  fname_ss << dir << global_myrank;
  printf("Writing file: %s\n", fname_ss.str().c_str());

  FILE * fp;
  if (binaryformat) {
    fp = fopen(fname_ss.str().c_str(), "wb");
    if (header) {
      auto fwrite_bytes = fwrite(&(edgelist.m), sizeof(int), 1, fp);
      assert(fwrite_bytes == 1);
      fwrite_bytes = fwrite(&(edgelist.n), sizeof(int), 1, fp);
      assert(fwrite_bytes == 1);
      fwrite_bytes = fwrite(&(edgelist.nnz), sizeof(int), 1, fp);
      assert(fwrite_bytes == 1);
    }
  } else {
    fp = fopen(fname_ss.str().c_str(), "w");
    if (header) {
      fprintf(fp, "%d %d %u\n", edgelist.m, edgelist.n, edgelist.nnz);
    }
  }
  for(auto i = 0 ; i < edgelist.nnz ; i++)
  {
    writeLine<T>(fp, edgelist.edges[i].src, edgelist.edges[i].dst, edgelist.edges[i].val, binaryformat, edgeweights);
  }
  fclose(fp);
}

template <typename T>
void load_edgelist(const char* dir, edgelist_t<T>* edgelist,
                             bool binaryformat=true, bool header=true, bool edgeweights=true) {
  int global_nrank = get_global_nrank();
  int global_myrank = get_global_myrank();
  edgelist->m = 0;
  edgelist->n = 0;
  edgelist->nnz = 0;
  for(int i = global_myrank ; ; i += global_nrank)
  {
    std::stringstream fname_ss;
    fname_ss << dir << i;
    FILE* fp;
    if (binaryformat) {
     fp = fopen(fname_ss.str().c_str(), "rb");
    } else {
     fp = fopen(fname_ss.str().c_str(), "r");
    }  
    if(!fp) {
      printf("Could not open file: %s\n", fname_ss.str().c_str());
      break;
    } else {
      printf("Reading file: %s\n", fname_ss.str().c_str());
    }

    int m_, n_;
    unsigned long nnz_;
    get_maxid_and_nnz<T>(fp, &m_, &n_, &nnz_, binaryformat, header, edgeweights);
    edgelist->m = std::max(m_, edgelist->m);
    edgelist->n = std::max(n_, edgelist->n);
    edgelist->nnz += nnz_;
    fclose(fp);
  }
  int local_max_m = edgelist->m;
  int max_m = edgelist->m;
  int local_max_n = edgelist->n;
  int max_n = edgelist->n;
  MPI_Allreduce(&local_max_m, &max_m, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max_n, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  edgelist->m = max_m;
  edgelist->n = max_n;

  std::cout << "Got: " << edgelist->m << " by " << edgelist->n << "  vertices" << std::endl;
  std::cout << "Got: " << edgelist->nnz << " edges" << std::endl;
  
  edgelist->edges = reinterpret_cast<edge_t<T>*>(
      _mm_malloc((uint64_t)edgelist->nnz * (uint64_t)sizeof(edge_t<T>), 64));


  unsigned long int nnzcnt = 0;
  for(int i = global_myrank ; ; i += global_nrank)
  {
    std::stringstream fname_ss;
    fname_ss << dir << i;
    //printf("Opening file: %s\n", fname_ss.str().c_str());
    FILE* fp;
    if (binaryformat) {
     fp = fopen(fname_ss.str().c_str(), "rb");
    } else {
     fp = fopen(fname_ss.str().c_str(), "r");
    }
    if(!fp) break;

    if (header) { //remove header
      int m_, n_;
      unsigned long nnz_;
      get_maxid_and_nnz<T>(fp, &m_, &n_, &nnz_, binaryformat, header, edgeweights);
    }
    int j = 0;
    while(true) {
      if (feof(fp)) {
        break;
      }
      if (!readLine<T>(fp, &(edgelist->edges[nnzcnt].src), &(edgelist->edges[nnzcnt].dst), &(edgelist->edges[nnzcnt].val), binaryformat, edgeweights)) {
        break;
      }
      #ifdef __DEBUG
      //std::cout <<(edgelist->edges[nnzcnt].src) << " " << (edgelist->edges[nnzcnt].dst) << std::endl;
      if(edgelist->edges[nnzcnt].src <= 0 ||
         edgelist->edges[nnzcnt].dst <= 0 ||
         edgelist->edges[nnzcnt].src > edgelist->m ||
         edgelist->edges[nnzcnt].dst > edgelist->n)
      {
        std::cout << "Invalid edge, i, j, nnz: " << i << " , " << j << " , " << nnzcnt << std::endl;
        exit(0);
      }
      j++;
      #endif
      nnzcnt++;
    }
    fclose(fp);
  }
}

template <typename T>
void randomize_edgelist_square(edgelist_t<T>* edgelist) {
  unsigned int* mapping = new unsigned int[edgelist->m];
  unsigned int* rval = new unsigned int[edgelist->m];

  int global_myrank = get_global_myrank();
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

template<typename T>
void get_dimensions(edge_t<T> * edges, int nnz, int &max_m, int &max_n)
{ 
  int local_max_m = 0;
  int local_max_n = 0;
  #pragma omp parallel for reduction(max:local_max_m, local_max_n)
  for(int i = 0 ; i < nnz ; i++)
  {
    local_max_m = std::max(local_max_m, edges[i].src);
    local_max_n = std::max(local_max_n, edges[i].dst);
  }
  MPI_Allreduce(&local_max_m, &max_m, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max_n, &max_n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
}

template <typename T>
void ReadEdges(edgelist_t<T>* edgelist, const char* fname_in, bool binaryformat=true, bool header=true, bool edgeweights=true, bool randomize=false) {
  load_edgelist(fname_in, edgelist, binaryformat, header, edgeweights);

  if (randomize) {
    randomize_edgelist_square<T>(edgelist);
  }
}

template <typename T>
void WriteEdges(const edgelist_t<T>& edgelist, const char* fname_in, bool binaryformat=true, bool header=true, bool edgeweights=true) {
  write_edgelist(fname_in, edgelist, binaryformat, header, edgeweights);
}

#endif  // SRC_EDGELIST_H_
