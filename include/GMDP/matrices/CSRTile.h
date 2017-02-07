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


#ifndef SRC_CSRTILE_H_
#define SRC_CSRTILE_H_

#include <string>
#include <algorithm>
#include <vector>

#include "GMDP/utils/binary_search.h"

template <typename T>
bool compare_notrans(const edge_t<T>& a, const edge_t<T>& b) {
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
class CSRTile {
 public:
  std::string name;
  int m;
  int n;
  int nnz;
  T* a;
  int* ja;
  int* ia;

  // Serialize
  friend boost::serialization::access;
  template<class Archive> 
  void save(Archive& ar, const unsigned int version) const {
    ar & name;
    ar & m;
    ar & n;
    ar & nnz;
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
      for(int i = 0 ; i < m+1 ; i++)
      {
        ar & ia[i];
      }
    }
  }

  template<class Archive> 
  void load(Archive& ar, const unsigned int version) {
    ar & name;
    ar & m;
    ar & n;
    ar & nnz;
    if(!isEmpty())
    {
      a = reinterpret_cast<T*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(T), 64));
      ja = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(int), 64));
      ia = reinterpret_cast<int*>(_mm_malloc((m + 1) * sizeof(int), 64));
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & a[i];
      }
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & ja[i];
      }
      for(int i = 0 ; i < m+1 ; i++)
      {
        ar & ia[i];
      }
    }
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()
 

  CSRTile() : name("TEMP"), m(0), n(0), nnz(0) {}

  CSRTile(int _m, int _n) : name("TEMP"), m(_m), n(_n), nnz(0) {}

  CSRTile(edge_t<T>* edges, int _m, int _n, int _nnz, int row_start,
          int col_start)
      : name("TEMP"), m(_m), n(_n), nnz(_nnz) {
      double stt = MPI_Wtime();
    if (nnz > 0) {
      __gnu_parallel::sort(edges, edges + nnz, compare_notrans<T>);
      a = reinterpret_cast<T*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(T), 64));
      ja = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(int), 64));
      int * jia  = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(int), 64));
      ia = reinterpret_cast<int*>(_mm_malloc((m + 1) * sizeof(int), 64));

      // convert to CSR
      #pragma omp parallel for
      for (uint64_t i = 0; i < (uint64_t)nnz; i++) {
        a[i] = edges[i].val;
        ja[i] = edges[i].dst - col_start; // one-based
        jia[i] = edges[i].src - row_start; // one-based
      }
      // Assign ia in parallel
      int num_partitions = omp_get_max_threads() * 4;
      int rows_per_partition = (m + num_partitions-1) / num_partitions;
      //#pragma omp parallel for
      for(int p = 0 ; p < num_partitions ; p++)
      {
        int start_row = rows_per_partition * p;
        int end_row = rows_per_partition * (p+1);
        if(start_row > m) start_row = m;
        if(end_row > m) end_row = m;

        // Find first 
        int nz_start = -1;
        int row = start_row ;
        while((row < m) && (nz_start == -1))
        {
          if(row+1 < jia[0])
          {
            nz_start = 0;
          }
          else if(row+1 > jia[nnz-1])
          {
            nz_start = nnz;
          }
          else
          {
            nz_start = binary_search_left_border(jia, row+1, 0, nnz, nnz);
          }
          row++;
        }
        if(nz_start == -1)
        {
          nz_start = nnz;
        }

        int current_row = start_row;
        ia[current_row] = nz_start;
        for (int i = nz_start; i < nnz; i++) {
          if(current_row >= end_row) break;
          while ((jia[i] > current_row) && (current_row < end_row)) {
            ia[current_row] = i + 1;
            current_row++;
          }
        }
        while (current_row < end_row) {
          ia[current_row] = nnz + 1;
          current_row++;
        }
      }
      ia[m] = nnz+1;
      _mm_free(jia);
    }
  }

  bool isEmpty() const { return nnz <= 0; }

  void get_edges(edge_t<T>* edges, int row_start, int col_start) {
    int nnzcnt = 0;
    if(this->nnz > 0)
    {
      #pragma omp parallel for
      for (int i = 0; i < this->m; i++) {
        for (int nz_id = ia[i]; nz_id < ia[i + 1]; nz_id++) {
          edges[nz_id-1].src = i + row_start + 1;
          edges[nz_id-1].dst = ja[nz_id - 1] + col_start;
          edges[nz_id-1].val = a[nz_id - 1];
        }
      }
      //assert(nnzcnt == this->nnz);
    }
  }

  CSRTile& operator=(CSRTile other) {
    this->name = other.name;
    this->m = other.m;
    this->n = other.n;
    this->nnz = other.nnz;
    this->a = other.a;
    this->ia = other.ia;
    this->ja = other.ja;
  }

  ~CSRTile(void) {
    if (!isEmpty()) {
      _mm_free(a);
      _mm_free(ja);
      _mm_free(ia);
    }
    nnz = 0;
  }
};

#endif  // SRC_CSRTILE_H_
