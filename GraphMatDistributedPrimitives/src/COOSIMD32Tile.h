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


#ifndef SRC_COOSIMD32TILE_H_
#define SRC_COOSIMD32TILE_H_

#include <string>
#include <algorithm>
#include <parallel/algorithm>
#include <vector>

#include "binary_search.h"

template <typename T>
bool compare_notrans_coosimd32(const tedge_t<T> & a, const tedge_t<T> & b)
{
  if(a.tile_id < b.tile_id) return true;
  else if(a.tile_id > b.tile_id) return false;

  if(a.dst < b.dst) return true;
  else if(a.dst > b.dst) return false;

  if(a.src < b.src) return true;
  else if(a.src > b.src) return false;

  return false;
}

struct partition_bin_coosimd32 {
  int p32;
  int num_total;
  int num_left;
  int num_taken;
};


template <typename T>
class COOSIMD32Tile {
 public:
  std::string name;
  int m;
  int n;
  int nnz;
  T* a;
  int* ja;
  int* ia;
  int * partition_start;
  int * simd_nnz;
  int num_partitions;

  COOSIMD32Tile() : name("TEMP"), m(0), n(0), nnz(0) {}

  COOSIMD32Tile(int _m, int _n) : name("TEMP"), m(_m), n(_n), nnz(0) {}

  COOSIMD32Tile(edge_t<T>* edges, int _m, int _n, int _nnz, int row_start,
          int col_start)
      : name("TEMP"), m(_m), n(_n), nnz(_nnz) {
      double stt = MPI_Wtime();
    if (nnz > 0) {
      a = reinterpret_cast<T*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(T), 64));
      ja = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(int), 64));
      ia = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)nnz * (uint64_t)sizeof(int), 64));
      tedge_t<T> * tmpedges = reinterpret_cast<tedge_t<T> *>( _mm_malloc(((uint64_t)nnz) * (uint64_t)sizeof(tedge_t<T>), 64));
      num_partitions = omp_get_max_threads() * 4;

      // Set partition IDs
      #pragma omp parallel for
      for(int edge_id = 0 ; edge_id < nnz ; edge_id++)
      {
        tmpedges[edge_id].src = edges[edge_id].src - row_start;
        tmpedges[edge_id].dst = edges[edge_id].dst - col_start;
        tmpedges[edge_id].val = edges[edge_id].val;
        tmpedges[edge_id].tile_id = (tmpedges[edge_id].src-1) / 32;
      }

      // Sort
      __gnu_parallel::sort(tmpedges, tmpedges+((uint64_t)nnz), compare_notrans_coosimd32<T>);

      #pragma omp parallel for
      for(int edge_id = 0 ; edge_id < nnz ; edge_id++)
      {
        ia[edge_id] = tmpedges[edge_id].src;
      }

      // Set partitions
      num_partitions = omp_get_max_threads() * 4;
      int rows_per_partition = (m + num_partitions - 1) / num_partitions;
      rows_per_partition = ((rows_per_partition + 31) / 32) * 32;
      partition_start = new int[num_partitions+1];
      simd_nnz = new int[num_partitions+1];
      
      int nnztotal = 0;
      #pragma omp parallel for 
      for (int p = 0; p < num_partitions ; p++) 
      {
        int start_row = p * rows_per_partition;
        int end_row = (p+1) * rows_per_partition;
        if(start_row > m) start_row = m;
        if(end_row > m) end_row = m;
        //int start_edge_id = l_binary_search(0, nnz, ia, start_row+1);
        //int end_edge_id = l_binary_search(0, nnz, ia, end_row+1);
        
        int start_edge_id = l_linear_search(0, nnz, ia, start_row+1);
        int end_edge_id = l_linear_search(0, nnz, ia, end_row+1);
        partition_start[p] = start_edge_id;
#ifdef __DEBUG
        assert(start_edge_id == l_linear_search(0, nnz, ia, start_row+1));
        assert(end_edge_id == l_linear_search(0, nnz, ia, end_row+1));
        assert(start_edge_id >= 0);
#endif
        int partition_nnz = end_edge_id - start_edge_id;
      }
      partition_start[num_partitions] = nnz;


      // Create arrays
      #pragma omp parallel for
      for (int p = 0; p < num_partitions; p++) 
      {
        int start_row = p * rows_per_partition;
        int end_row = (p+1) * rows_per_partition;
        if(start_row > m) start_row = m;
        if(end_row > m) end_row = m;
        int start_edge_id = partition_start[p];
        int end_edge_id = partition_start[p+1];
        int partition_nnz = end_edge_id - start_edge_id;

        // For each 32 partition
        int npartitions = ((end_row-start_row) + 31) / 32;
        int * borders = new int[npartitions+1];
        int current_partition = 0;
        for(int eid = start_edge_id ; eid < end_edge_id ; eid++)
        {
          int new_partition = (ia[eid] - start_row - 1) / 32;
          while(current_partition <= new_partition)
          {
            borders[current_partition] = eid;
            current_partition++;
          }
        }
        while(current_partition <= npartitions)
        {
          borders[current_partition] = end_edge_id;
          current_partition++;
        }

        std::vector<partition_bin_coosimd32> bins = std::vector<partition_bin_coosimd32>(npartitions);
        int n_full32 = 0;
        for(int bin = 0 ; bin < npartitions ; bin++)
        {
          bins[bin].p32 = bin;
          bins[bin].num_total = bins[bin].num_left = borders[bin+1]-borders[bin];
          bins[bin].num_taken = 0;
          if(bins[bin].num_total > 0) n_full32++;
        }

        // Sort bins for heuristic
        std::sort(bins.begin(), bins.end(), 
          [](partition_bin_coosimd32 const & bin1, partition_bin_coosimd32 const & bin2) -> bool { return bin1.num_total > bin2.num_total; });

        // Round robin assignment
        int nnzsimd = 0;
        while(n_full32 >= 32)
        {
          int rotation_count = 0;
          for(int bin = 0 ; bin < npartitions ; bin++)
          {
            if(bins[bin].num_left > 0) 
            {
              int eid = borders[bins[bin].p32] + bins[bin].num_taken;
              bins[bin].num_taken++;
              bins[bin].num_left--;
              if(bins[bin].num_left == 0) n_full32--;

              // Copy edge
              ja[nnzsimd+partition_start[p]] = tmpedges[eid].dst;
              ia[nnzsimd+partition_start[p]] = tmpedges[eid].src;
              a[nnzsimd+partition_start[p]] = tmpedges[eid].val;

              nnzsimd++;
              rotation_count++;
            }
            if(rotation_count == 32) break;
          }
        }
        simd_nnz[p] = nnzsimd;

        int nnzincrement = nnzsimd;
        for(int bin = 0 ; bin < npartitions ; bin++)
        {
          for(int taken_cnt = bins[bin].num_taken ; taken_cnt < bins[bin].num_total ; taken_cnt++)
          {
            int eid = borders[bins[bin].p32] + taken_cnt;

            // Copy edge
            ja[nnzincrement+partition_start[p]] = tmpedges[eid].dst;
            ia[nnzincrement+partition_start[p]] = tmpedges[eid].src;
            a[nnzincrement+partition_start[p]] = tmpedges[eid].val;
            nnzincrement++;
          }
        }

        if(nnzincrement != partition_nnz)
        {
          std::cout << "nnzincrement: " << nnzincrement << "\t partition_nnz: " << partition_nnz << std::endl;
          exit(0);
        }
        assert(nnzincrement == (partition_start[p+1]-partition_start[p]));

        delete [] borders;
      }

#ifdef __DEBUG
      unsigned int total_nnz_simd = 0;
      for(int p = 0 ; p < num_partitions; p++)
      {
        total_nnz_simd += simd_nnz[p];
      }
      std::cout << "COOSIMD32 SIMD precentage: " << ((double) total_nnz_simd) / ((double)nnz) << std::endl;
      // Check against edgelist
      tedge_t<T> * check_edges = new tedge_t<T>[nnz];
      for(int nzid = 0 ; nzid < nnz ; nzid++)
      {
        check_edges[nzid].dst = ja[nzid];
        check_edges[nzid].src = ia[nzid];
        check_edges[nzid].val = a[nzid];
        check_edges[nzid].tile_id = (ia[nzid]-1) / 32;
      }
      __gnu_parallel::sort(check_edges, check_edges+nnz, compare_notrans_coosimd32<T>);

      #pragma omp parallel
      for(int i = 0 ; i < nnz ; i++)
      {
        assert(tmpedges[i].dst == check_edges[i].dst);
        assert(tmpedges[i].src == check_edges[i].src);
        //assert(tmpedges[i].val == check_edges[i].val); // commented now in case of duplicate edges
      }

      delete [] check_edges;

      #pragma omp parallel for
      for(int p = 0 ; p < num_partitions ; p++) {
        assert(simd_nnz[p] % 32 == 0);
        assert(simd_nnz[p] <= (partition_start[p+1] - partition_start[p]));
        for(int i32 = 0 ; i32 < simd_nnz[p] ; i32+=32) {
          for(int i = 0 ; i < 32 ; i++) {
            for(int j = i+1 ; j < 32 ; j++) {
              int partition_i = (ia[partition_start[p] + i32 + i]-1) / 32;
              int partition_j = (ia[partition_start[p] + i32 + j]-1) / 32;
              assert(partition_i != partition_j);
            }
          }
        }
      }
#endif // __DEBUG

      //delete [] tmpedges;
      _mm_free(tmpedges);
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

  COOSIMD32Tile& operator=(COOSIMD32Tile other) {
    this->name = other.name;
    this->m = other.m;
    this->n = other.n;
    this->nnz = other.nnz;
    this->a = other.a;
    this->ia = other.ia;
    this->ja = other.ja;
    this->num_partitions = other.num_partitions;
    this->partition_start = other.partition_start;
    this->simd_nnz = other.simd_nnz;
  }

  void clear() {
    if (!isEmpty()) {
      _mm_free(a);
      _mm_free(ja);
      _mm_free(ia);
      delete [] partition_start;
    }
    nnz = 0;
  }

  ~COOSIMD32Tile(void) {}

  void send_tile_metadata(int myrank, int dst_rank, int output_rank) {
    assert(0);
  }

  void recv_tile_metadata(int myrank, int src_rank, int output_rank) {
    assert(0);
  }

  void send_tile(int myrank, int dst_rank, int output_rank, bool block, std::vector<MPI_Request>* reqs) {
    assert(0);
  }

  void recv_tile(int myrank, int src_rank, int output_rank, bool block,
                 std::vector<MPI_Request>* reqs) {
    assert(0);
  }
};

#endif  // SRC_COOSIMD32TILE_H_
