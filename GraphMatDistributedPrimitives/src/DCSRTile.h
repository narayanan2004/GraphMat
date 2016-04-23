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


#ifndef SRC_DCSRTILE_H_
#define SRC_DCSRTILE_H_

#include <string>
#include <algorithm>
#include <vector>

#include "binary_search.h"

template <typename T>
class DCSRTile {
 public:
  std::string name;
  int m;
  int n;
  int num_rows;
  int nnz;
  int num_partitions;
  int * partition_ptrs;
  T* a;
  int* ja;
  int* ia;
  int* row_ids;

  DCSRTile() : name("TEMP"), m(0), n(0), nnz(0) {}

  DCSRTile(int _m, int _n) : name("TEMP"), m(_m), n(_n), nnz(0) {}

  DCSRTile(edge_t<T>* edges, int _m, int _n, int _nnz, int row_start,
          int col_start)
      : name("TEMP"), m(_m), n(_n), nnz(_nnz) {
    if(nnz > 0)
    {
      __gnu_parallel::sort(edges, edges + nnz, [](const edge_t<T>& a, const edge_t<T>& b) 
      {
        if (a.src < b.src) return true; else if (a.src > b.src) return false;
        if (a.dst < b.dst) return true; else if (a.dst > b.dst) return false;
        return false;
      });

      int * tmp_buf = new int[nnz];
      tmp_buf[0] = 0;
      for(int i = 0 ; i < nnz-1 ; i++)
      {
        if(edges[i+1].src > edges[i].src)
        {
          tmp_buf[i+1] = tmp_buf[i] + 1;
        }
        else
        {
          tmp_buf[i+1] = tmp_buf[i];
        }
      }
      num_rows = tmp_buf[nnz-1]+1;

      row_ids = reinterpret_cast<int*>(_mm_malloc(((num_rows)) * sizeof(int), 64));
      ia = reinterpret_cast<int*>(_mm_malloc(((num_rows) + 1) * sizeof(int), 64));
      
      row_ids[0] = (edges[0].src - row_start) - 1;
      ia[0] = 0;
      for(int i = 0 ; i < nnz-1 ; i++)
      {
        if(edges[i+1].src > edges[i].src)
        {
          row_ids[tmp_buf[i+1]] = (edges[i+1].src - row_start) - 1;
          ia[tmp_buf[i+1]] = i+1;
        }
      }
      ia[num_rows] = nnz;
      delete [] tmp_buf;

      num_partitions = omp_get_max_threads() * 4;
      partition_ptrs = new int[num_partitions+1];
      int rows_per_partition = ((num_rows + num_partitions) - 1) / num_partitions;
      partition_ptrs[0] = 0;
      for(int p = 1 ; p < num_partitions ; p++)
      {
        int new_row = partition_ptrs[p-1] + rows_per_partition;
        if(new_row > num_rows)
        {
          new_row = num_rows;
        }

        // Increase new row to next 32-bit boundary
        int row32 = row_ids[new_row] / 32;
        while((new_row < num_rows) && ((row_ids[new_row] / 32) == row32))
        {
          new_row++;
        }
        partition_ptrs[p] = new_row;
      }
      partition_ptrs[num_partitions] = num_rows;

      ja = reinterpret_cast<int*>(_mm_malloc((nnz ) * sizeof(int), 64));
      a = reinterpret_cast<T*>(_mm_malloc((nnz) * sizeof(T), 64));

      for(int i = 0 ; i < num_rows ; i++)
      {
        for(int j = ia[i] ; j < ia[i+1] ; j++)
        {
          ja[j] = (edges[j].dst - col_start) - 1;
          a[j] = edges[j].val;
        }
      }

#ifdef __DEBUG
      unsigned long int nzcnt = 0;
      for(int p = 0 ; p < num_partitions ; p++)
      {
        for(int _row = partition_ptrs[p] ; _row < partition_ptrs[p+1]; _row++)
        {
          int row = row_ids[_row];
          for(int j = ia[_row] ; j < ia[_row+1] ; j++)
          {
            assert(edges[nzcnt].src == (row + row_start + 1) );
            assert(edges[nzcnt].dst == (ja[j] + col_start + 1));
            assert(edges[nzcnt].val == (a[j]));
            nzcnt++;
          }
        }
      }
      assert(nzcnt == nnz);
#endif
    }
  }

  bool isEmpty() const { return nnz <= 0; }

  void get_edges(edge_t<T>* edges, int row_start, int col_start) {
    unsigned int nnzcnt = 0;
    if(this->nnz > 0)
    {
      #pragma omp parallel for reduction(+:nnzcnt)
      for (int i = 0; i < this->num_rows; i++) {
        for (int nz_id = ia[i]; nz_id < ia[i + 1]; nz_id++) {
          edges[nz_id].src = row_ids[i] + row_start + 1; 
          edges[nz_id].dst = ja[nz_id] + col_start + 1;
          edges[nz_id].val = a[nz_id];
          nnzcnt++;
        }
      }
      assert(nnzcnt == this->nnz);
    }
  }

  DCSRTile& operator=(DCSRTile other) {
    this->name = other.name;
    this->m = other.m;
    this->n = other.n;
    this->num_rows = other.num_rows;
    this->nnz = other.nnz;
    this->a = other.a;
    this->ia = other.ia;
    this->row_ids = other.row_ids;
    this->ja = other.ja;
    this->num_partitions = other.num_partitions;
    this->partition_ptrs = other.partition_ptrs;
  }

  void clear() {
    if (!isEmpty()) {
      _mm_free(a);
      _mm_free(ja);
      _mm_free(ia);
      _mm_free(row_ids);
      delete [] partition_ptrs;
    }
    nnz = 0;
  }

  ~DCSRTile(void) {}

};

#endif  // SRC_DCSRTILE_H_
