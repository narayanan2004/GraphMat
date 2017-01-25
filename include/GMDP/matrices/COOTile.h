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


#ifndef SRC_COOTILE_H_
#define SRC_COOTILE_H_

#include <string>
#include <algorithm>
#include <vector>

#include "GMDP/utils/binary_search.h"

template <typename T>
bool compare_notrans_coo(const edge_t<T>& a, const edge_t<T>& b) {
  if (a.src < b.src)
    return true;
  else if (a.src > b.src)
    return false;

  if (a.dst < b.dst)
    return true;
  else if (a.dst > b.dst)
    return false;
  return false;
}

template <typename T>
class COOTile {
 public:
  std::string name;
  int m;
  int n;
  int nnz;
  int num_partitions;
  T* a;
  int* ja;
  int* ia;
  int * partition_start; // num_partitions+1

  // Serialize
  friend boost::serialization::access;
  template<class Archive> 
  void save(Archive& ar, const unsigned int version) const {
    ar & name;
    ar & m;
    ar & n;
    ar & nnz;
    ar & num_partitions;
    if(!isEmpty())
    {
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & a[i];
      }
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & ja[i];
      }
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & ia[i];
      }
      for(int i = 0 ; i < num_partitions+1 ; i++)
      {
        ar & partition_start[i];
      }
    }
  }

  template<class Archive> 
  void load(Archive& ar, const unsigned int version) {
    ar & name;
    ar & m;
    ar & n;
    ar & nnz;
    ar & num_partitions;
    if(!isEmpty())
    {
      a = reinterpret_cast<T*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(T), 64));
      ja = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(int), 64));
      ia = reinterpret_cast<int*>(_mm_malloc(nnz * sizeof(int), 64));
      partition_start  = reinterpret_cast<int*>(_mm_malloc((num_partitions+1) * sizeof(int), 64));
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & a[i];
      }
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & ja[i];
      }
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & ia[i];
      }
      for(int i = 0 ; i < num_partitions+1 ; i++)
      {
        ar & partition_start[i];
      }
    }
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()
 

  COOTile() : name("TEMP"), m(0), n(0), nnz(0) {}

  COOTile(int _m, int _n) : name("TEMP"), m(_m), n(_n), nnz(0) {}

  COOTile(edge_t<T>* edges, int _m, int _n, int _nnz, int row_start,
          int col_start)
      : name("TEMP"), m(_m), n(_n), nnz(_nnz) {
      double stt = MPI_Wtime();
    if (nnz > 0) {
      __gnu_parallel::sort(edges, edges + nnz, compare_notrans_coo<T>);
      a = reinterpret_cast<T*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(T), 64));
      ja = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(int), 64));
      ia = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(int), 64));

      // convert to COO
      #pragma omp parallel for
      for (uint64_t i = 0; i < (uint64_t)nnz; i++) {
        a[i] = edges[i].val;
        ja[i] = edges[i].dst - col_start; // one-based
        ia[i] = edges[i].src - row_start; // one-based
#ifdef __DEBUG
        assert(ja[i] > 0);
        assert(ja[i] <= n);
        assert(ia[i] > 0);
        assert(ia[i] <= m);
#endif
      }

      // Set partitions
      num_partitions = omp_get_max_threads() * 4;
      int rows_per_partition = (m + num_partitions - 1) / num_partitions;
      rows_per_partition = ((rows_per_partition + 31) / 32) * 32;
      partition_start = new int[num_partitions+1];
      
      #pragma omp parallel for
      for(int p = 0 ; p < num_partitions ; p++)
      {
        int start_row = p * rows_per_partition;
        int end_row = (p+1) * rows_per_partition;
        if(start_row > m) start_row = m;
        if(end_row > m) end_row = m;
        int start_edge_id = l_binary_search(0, nnz, ia, start_row+1);
        int end_edge_id = l_binary_search(0, nnz, ia, end_row+1);
        partition_start[p] = start_edge_id;
#ifdef __DEBUG
        assert(start_edge_id == l_linear_search(0, nnz, ia, start_row+1));
        assert(end_edge_id == l_linear_search(0, nnz, ia, end_row+1));
        assert(start_edge_id >= 0);
#endif
      }
      partition_start[num_partitions] = nnz;
    }
  }

  bool isEmpty() const { return nnz <= 0; }

  void get_edges(edge_t<T>* edges, int row_start, int col_start) {
    int nnzcnt = 0;
    #pragma omp parallel for
    for (uint64_t i = 0; i < (uint64_t)nnz; i++) {
      edges[i].val = a[i];
      edges[i].dst = ja[i] + col_start;
      edges[i].src = ia[i] + row_start;
    }
  }

  COOTile& operator=(COOTile other) {
    this->name = other.name;
    this->m = other.m;
    this->n = other.n;
    this->nnz = other.nnz;
    this->a = other.a;
    this->ia = other.ia;
    this->ja = other.ja;
    this->num_partitions = other.num_partitions;
    this->partition_start = other.partition_start;
  }

  void clear() {
  }

  ~COOTile(void) {
    if (!isEmpty()) {
      _mm_free(a);
      _mm_free(ja);
      _mm_free(ia);
      delete [] partition_start;
    }
    nnz = 0;
}

};

#endif  // SRC_COOTILE_H_
