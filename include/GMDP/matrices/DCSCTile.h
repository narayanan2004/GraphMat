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


#ifndef SRC_DCSCTILE_H_
#define SRC_DCSCTILE_H_

#include <string>
#include <algorithm>
#include <vector>

template <typename T>
bool compare_dcsc(const tedge_t<T> &a, const tedge_t<T> &b) {
  if (a.tile_id < b.tile_id)
    return true;
  else if (a.tile_id > b.tile_id)
    return false;

  if (a.dst < b.dst)
    return true;
  else if (a.dst > b.dst)
    return false;

  if (a.src < b.src)
    return true;
  else if (a.src > b.src)
    return false;

  return false;
}

template <typename T>
class DCSCTile {
 public:
  int nnz;
  std::string name;
  int m;
  int n;
  int num_cols;
  int num_partitions;

  T *vals;        // nnz
  int *row_inds;  // nnz
  int *row_pointers; // num_partitions + 1
  int *edge_pointers;  // num_partitions + 1
  int *col_starts; // num_partitions + 1
  int *col_ptrs;  // col_starts[num_partitions]
  int *col_indices; // col_starts[num_partitions]

  // Serialize
  friend boost::serialization::access;
  template<class Archive> 
  void save(Archive& ar, const unsigned int version) const {
    ar & nnz;
    ar & name;
    ar & m;
    ar & n;
    ar & num_cols;
    ar & num_partitions;
    if(nnz > 0)
    {
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & vals[i];
      }
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & row_inds[i];
      }
      for(int i = 0 ; i < num_partitions+1 ; i++)
      {
        ar & row_pointers[i];
      }
      for(int i = 0 ; i < num_partitions+1 ; i++)
      {
        ar & edge_pointers[i];
      }
      for(int i = 0 ; i < num_partitions+1 ; i++)
      {
        ar & col_starts[i];
      }
      for(int i = 0 ; i < col_starts[num_partitions] ; i++)
      {
        ar & col_ptrs[i];
      }
      for(int i = 0 ; i < col_starts[num_partitions] ; i++)
      {
        ar & col_indices[i];
      }
    }
  }

  template<class Archive> 
  void load(Archive& ar, const unsigned int version) {
    ar & nnz;
    ar & name;
    ar & m;
    ar & n;
    ar & num_cols;
    ar & num_partitions;

    std::cout << "Got nnz: " << nnz << std::endl;

    if(nnz > 0)
    {
      vals = reinterpret_cast<T *>(
          _mm_malloc(nnz * sizeof(T), 64));
      row_inds = reinterpret_cast<int *>(
          _mm_malloc(nnz * sizeof(int), 64));
      row_pointers = reinterpret_cast<int *>(
          _mm_malloc((num_partitions+1) * sizeof(int), 64));
      edge_pointers = reinterpret_cast<int *>(
          _mm_malloc((num_partitions+1) * sizeof(int), 64));
      col_starts = reinterpret_cast<int *>(
          _mm_malloc((num_partitions+1) * sizeof(int), 64));
      col_ptrs = reinterpret_cast<int *>(
          _mm_malloc(num_cols * sizeof(int), 64));
      col_indices = reinterpret_cast<int *>(
          _mm_malloc(num_cols * sizeof(int), 64));

      for(int i = 0 ; i < nnz ; i++)
      {
        ar & vals[i];
      }
      for(int i = 0 ; i < nnz ; i++)
      {
        ar & row_inds[i];
      }
      for(int i = 0 ; i < num_partitions+1 ; i++)
      {
        ar & row_pointers[i];
      }
      for(int i = 0 ; i < num_partitions+1 ; i++)
      {
        ar & edge_pointers[i];
      }
      for(int i = 0 ; i < num_partitions+1 ; i++)
      {
        ar & col_starts[i];
      }
      for(int i = 0 ; i < num_cols ; i++)
      {
        ar & col_ptrs[i];
      }
      for(int i = 0 ; i < num_cols ; i++)
      {
        ar & col_indices[i];
      }
    }
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  
  DCSCTile() : name("TEMP"), m(0), n(0), nnz(0), num_partitions(0) {}

  DCSCTile(int _m, int _n) : name("TEMP"), m(_m), n(_n), nnz(0), num_partitions(0) {}

  static void static_partition(int *&row_pointers, int m, int num_partitions,
                               int round) {
    row_pointers = reinterpret_cast<int *>(
        _mm_malloc((num_partitions + 1) * sizeof(int), 64));

    if (round == 1) {
      int rows_per_partition = m / num_partitions;
      int rows_leftover = m % num_partitions;
      row_pointers[0] = 0;
      int current_row = row_pointers[0] + rows_per_partition;
      for (int p = 1; p < num_partitions + 1; p++) {
        if (rows_leftover > 0) {
          current_row += 1;
          row_pointers[p] = current_row;
          current_row += rows_per_partition;
          rows_leftover--;
        } else {
          row_pointers[p] = current_row;
          current_row += rows_per_partition;
        }
      }
    } else {
      int n512 = std::max((m / round) / num_partitions, 1);
      int n_round = std::max(0, m / round - n512 * num_partitions);
      assert(n_round < num_partitions);
      row_pointers[0] = 0;
      for (int p = 1; p < num_partitions; p++) {
        row_pointers[p] =
            row_pointers[p - 1] +
            ((n_round > 0) ? ((n512 + 1) * round) : (n512 * round));
        row_pointers[p] = std::min(row_pointers[p], m);
        if (n_round > 0) n_round--;
      }
      row_pointers[num_partitions] = m;
    }
  }

  static void set_edge_pointers(tedge_t<T> *edges, int *row_pointers,
                                int **edge_pointers, int nnz,
                                int num_partitions) {
    // Figure out edge pointers
    (*edge_pointers) = reinterpret_cast<int *>(
        _mm_malloc((num_partitions + 1) * sizeof(int), 64));
    int p = 0;
    for (int edge_id = 0; edge_id < nnz; edge_id++) {
      while (edges[edge_id].src >= row_pointers[p]) {
        (*edge_pointers)[p] = edge_id;
        p++;
      }
    }
    (*edge_pointers)[p] = nnz;
    for (p = p + 1; p < num_partitions + 1; p++) {
      (*edge_pointers)[p] = nnz;
    }
  }

  DCSCTile(edge_t<T> *edges, int _m, int _n, int _nnz, int row_start,
           int col_start)
      : name("TEMP"), m(_m), n(_n), nnz(_nnz) {

      double _start_time = MPI_Wtime();
    if (nnz > 0) {
      num_partitions = omp_get_max_threads() * 16;
      // Partition
      DCSCTile<T>::static_partition(row_pointers, this->m, num_partitions, 32);

      // Set partition IDs for each edge
      tedge_t<T> *p_edges = reinterpret_cast<tedge_t<T> *>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(tedge_t<T>), 64));

      std::cout << "num partitions: " << num_partitions << std::endl;
      double _ep_start = MPI_Wtime();
#pragma omp parallel for
      for (int i = 0; i < nnz; i++) {
        p_edges[i].src = edges[i].src - 1 - row_start;
        p_edges[i].dst = edges[i].dst - 1 - col_start;
        p_edges[i].val = edges[i].val;
        p_edges[i].tile_id = -1;

	int p_start = 0;
	int p_end = num_partitions-1;
	while(1)
	{
	  int p_half = p_start + (p_end - p_start);

	  // Check p_half
          if (p_edges[i].src >= row_pointers[p_half] &&
              p_edges[i].src < row_pointers[p_half + 1]) {
            p_edges[i].tile_id = p_half;
            break;
          }
	  if(p_edges[i].src < row_pointers[p_half]) p_end = p_half - 1;
	  if(p_edges[i].src >= row_pointers[p_half+1]) p_start = p_half + 1;
	}

#ifdef CHECK_BINARY_SEARCH
        for (int p = 0; p < num_partitions; p++) {
          if (p_edges[i].src >= row_pointers[p] &&
              p_edges[i].src < row_pointers[p + 1]) {
            //p_edges[i].tile_id = p;
            //break;
	    assert(p_edges[i].tile_id == p);
          }
#endif
        assert(p_edges[i].tile_id >= 0);
      }
      double _ep_end = MPI_Wtime();
      std::cout << "set_edge_pointers time: " << _ep_end - _ep_start << std::endl;

      // Sort
//      std::cout << "Sorting: " << (uint64_t)nnz << std::endl;
 //     std::cout << "allocated : " << (uint64_t)nnz * (uint64_t)sizeof(tedge_t<T>) << std::endl;
      #pragma omp parallel for
      for(int i =0 ; i < nnz ; i++)
      {
        assert(p_edges[i].src >= 0);
        assert(p_edges[i].dst >= 0);
        assert(p_edges[i].src < _m);
        assert(p_edges[i].dst < _n);
      }
      __gnu_parallel::sort(p_edges, p_edges + nnz, compare_dcsc<T>);

      // Find edge pointers
      DCSCTile<T>::set_edge_pointers(p_edges, row_pointers, &edge_pointers, nnz,
                                     num_partitions);

      // Count columns
      int *ncols =
          reinterpret_cast<int *>(_mm_malloc(num_partitions * sizeof(int), 64));
      col_starts = reinterpret_cast<int *>(
          _mm_malloc((num_partitions + 1) * sizeof(int), 64));
#pragma omp parallel for
      for (int p = 0; p < num_partitions; p++) {
        int current_column = -1;
        int num_columns = 0;
        for (int edge_id = edge_pointers[p]; edge_id < edge_pointers[p + 1];
             edge_id++) {
          if (current_column < p_edges[edge_id].dst) {
            num_columns++;
            current_column = p_edges[edge_id].dst;
          }
        }
        ncols[p] = num_columns;
      }

      int total_cols = 0;
      for (int p = 0; p < num_partitions; p++) {
        col_starts[p] = total_cols;
        total_cols += ncols[p] + 1;
      }
      col_starts[num_partitions] = total_cols;
      num_cols = total_cols;

      // Build DCSC
      std::cout << "Allocating nnz vals: " << nnz << std::endl;
      vals = reinterpret_cast<T *>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(T), 64));
      row_inds = reinterpret_cast<int *>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(int), 64));
      col_indices = reinterpret_cast<int *>(
          _mm_malloc(col_starts[num_partitions] * sizeof(int), 64));
      col_ptrs = reinterpret_cast<int *>(
          _mm_malloc(col_starts[num_partitions] * sizeof(int), 64));

#pragma omp parallel for
      for (int p = 0; p < num_partitions; p++) {
        T *val = vals + edge_pointers[p];
        int *row_ind = row_inds + edge_pointers[p];
        int *col_index = col_indices + col_starts[p];
        int *col_ptr = col_ptrs + col_starts[p];
        int current_column = -1;
        int current_column_num = -1;
        for (int edge_id = edge_pointers[p]; edge_id < edge_pointers[p + 1];
             edge_id++) {
          val[edge_id - edge_pointers[p]] = p_edges[edge_id].val;
          row_ind[edge_id - edge_pointers[p]] = p_edges[edge_id].src;
          if (current_column < p_edges[edge_id].dst) {
            current_column_num++;
            current_column = p_edges[edge_id].dst;
            col_index[current_column_num] = current_column;
            col_ptr[current_column_num] = edge_id - edge_pointers[p];
          }
        }
        int num_columns = col_starts[p + 1] - col_starts[p] - 1;
        col_ptr[num_columns] = edge_pointers[p + 1] - edge_pointers[p];
        col_index[num_columns] = n + 1;
      }
      _mm_free(p_edges);
      _mm_free(ncols);
    }
    else
    {
      num_partitions = 0;
    }
    double _end_time = MPI_Wtime();
    std::cout << "fn time: " << _end_time - _start_time << std::endl;
  }

  void get_edges(edge_t<T> *edges, int row_start, int col_start) {
    int nnzcnt = 0;
    for (int p = 0; p < num_partitions; p++) {
      for (int j = 0; j < (col_starts[p + 1] - col_starts[p]) - 1; j++) {
        int col_index = col_indices[col_starts[p] + j];
        for (int nz_idx = col_ptrs[col_starts[p] + j];
             nz_idx < col_ptrs[col_starts[p] + j + 1]; nz_idx++) {
          int row_ind = row_inds[edge_pointers[p] + nz_idx];
          edges[nnzcnt].src = row_start + row_ind + 1;
          edges[nnzcnt].dst = col_start + col_index + 1;
          edges[nnzcnt].val = vals[edge_pointers[p] + nz_idx];
          nnzcnt++;
        }
      }
    }
    assert(nnzcnt == this->nnz);
  }

  bool isEmpty() const { return nnz <= 0; }

  ~DCSCTile() 
  {
    if(nnz > 0)
    {
      _mm_free(row_inds);
      _mm_free(col_ptrs);
      _mm_free(col_indices);
      _mm_free(vals);
      _mm_free(row_pointers);
      _mm_free(edge_pointers);
      _mm_free(col_starts);
    }
    nnz = 0;
  }

};

#endif  // SRC_DCSCTILE_H_
