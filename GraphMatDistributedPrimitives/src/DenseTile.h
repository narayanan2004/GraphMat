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


#ifndef SRC_DENSETILE_H_
#define SRC_DENSETILE_H_

#include <string>
#include <vector>
#include "src/bitvector.h"

template <typename T>
class DenseTile {
 public:
  std::string name;
  int m;
  int n;
  int nnz;
  int empty_flag;
  int num_ints;
  int * bit_vector;
  T* value;

  DenseTile() : name("TEMP"), m(0), n(0), nnz(0), empty_flag(true), num_ints(0) {}

  DenseTile(int _m, int _n) : name("TEMP"), m(_m), n(_n), nnz(0), empty_flag(true) {
    num_ints = (_m * _n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
  }

  DenseTile(edge_t<T>* edges, int _m, int _n, int _nnz, int row_start,
            int col_start)
      : name("TEMP"), m(_m), n(_n), nnz(_nnz), empty_flag(false) {

    num_ints = (_m * _n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
    alloc();
    // convert to Dense
    for (uint64_t i = 0; i < (uint64_t)nnz; i++) {
      int src = edges[i].src - 1;
      int dst = edges[i].dst - 1;
      set_bitvector(src + dst * m, bit_vector);
      value[src + dst * m] = edges[i].val;
    }
  }

  void set(int idx, int idy, T val)
  {
    if(isEmpty()) {
      alloc();
    }
    value[(idy-1) + (idx-1) * m] = val;
    if(!get_bitvector((idy-1) + (idx-1) * m, bit_vector)) nnz++;
    empty_flag = false;
    set_bitvector((idy-1) + (idx-1) * m, bit_vector);
  }

  T get(int idx, int idy)
  {
    assert(!isEmpty());
    return value[(idy-1) + (idx-1) * m];
  }

  void alloc() {
    value = reinterpret_cast<T*>(
        _mm_malloc((uint64_t)(m * n) * (uint64_t)sizeof(T), 64));
    bit_vector = reinterpret_cast<int*>(
        _mm_malloc((uint64_t)(num_ints) * (uint64_t)sizeof(int), 64));
    memset(bit_vector, 0, (num_ints * sizeof(int)));
    memset(value, 0, (m * n * sizeof(T)));
    empty_flag = false;
    nnz = 0;
  }

  bool isEmpty() const { return empty_flag; }

  void get_edges(edge_t<T>* edges, int row_start, int col_start) {
    int nnzcnt = 0;
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        if (get_bitvector(i + j * m, bit_vector)) {
          edges[nnzcnt].src = i + 1;
          edges[nnzcnt].dst = j + 1;
          edges[nnzcnt].val = value[i + j * m];
          nnzcnt++;
        }
      }
    }
    assert(nnzcnt == this->nnz);
  }

  DenseTile& operator=(DenseTile other) {
    this->m = other.m;
    this->n = other.n;
    this->nnz = other.nnz;
    this->empty_flag = other.empty_flag;
    this->value = other.value;
    this->bit_vector = other.bit_vector;
    this->num_ints = other.num_ints;
  }

  void clear() {
    if (!isEmpty()) {
      _mm_free(value);
      _mm_free(bit_vector);
    }
    nnz = 0;
    empty_flag = true;
  }

  ~DenseTile() {}

  void send_tile_metadata(int myrank, int dst_rank, int output_rank) {
    MPI_Send(&(nnz), 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&(m), 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&(n), 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&(num_ints), 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
  }

  void recv_tile_metadata(int myrank, int src_rank, int output_rank) {
    int new_nnz;
    MPI_Recv(&(new_nnz), 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(&(m), 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(n), 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(num_ints), 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (isEmpty()) {
      value = reinterpret_cast<T*>(
          _mm_malloc((uint64_t)(m * n) * (uint64_t)sizeof(T), 64));
      bit_vector = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)(num_ints) * (uint64_t)sizeof(int), 64));
      memset(bit_vector, 0, (num_ints * sizeof(int)));
      empty_flag=false;
    }
    nnz = new_nnz;
  }

  void send_tile(int myrank, int dst_rank, int output_rank, bool block, std::vector<MPI_Request>* reqs) {
    block = true;
    if (!isEmpty()) {
    /*
      // Convert to edgelist
      edge_t<T>* edges = new edge_t<T>[nnz];
      int nzcnt = 0;
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          if (get_bitvector(i + j * m, bit_vector)) {
            edges[nzcnt].src = i + 1;
            edges[nzcnt].dst = j + 1;
            edges[nzcnt].val = value[i + j * m];
            nzcnt++;
          }
        }
      }
      */
      //assert(nzcnt == nnz);
      if (block) {
        //MPI_Send(edges, (uint64_t)nnz * sizeof(edge_t<T>), MPI_BYTE, dst_rank,
        //         0, MPI_COMM_WORLD);
         MPI_Send(value, (uint64_t) (m * n * sizeof(T)), MPI_BYTE,
         dst_rank, 0, MPI_COMM_WORLD);
         MPI_Send(bit_vector, (uint64_t) (num_ints) * sizeof(int),
         MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD);
      } else {
        MPI_Request r1, r2;
        //MPI_Isend(edges, (uint64_t)nnz * sizeof(edge_t<T>), MPI_BYTE, dst_rank,
        //          0, MPI_COMM_WORLD, &r1);
         MPI_Isend(value, (uint64_t) (m * n * sizeof(T)), MPI_BYTE,
         dst_rank, 0, MPI_COMM_WORLD, &r1);
         MPI_Isend(bit_vector, (uint64_t) (num_ints) * sizeof(int),
         MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD, &r2);
        (*reqs).push_back(r1);
        (*reqs).push_back(r2);
      }
      //delete[] edges;
    }
  }

  void recv_tile(int myrank, int src_rank, int output_rank, bool block,
                 std::vector<MPI_Request>* reqs) {
    block = true;
    if (!isEmpty()) {
      edge_t<T>* edges = new edge_t<T>[nnz];
      if (block) {
        //MPI_Recv(edges, (uint64_t)nnz * sizeof(edge_t<T>), MPI_BYTE, src_rank,
        //         0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         MPI_Recv(value, (uint64_t) (m * n * sizeof(T)), MPI_BYTE,
         src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         MPI_Recv(bit_vector, (uint64_t) (num_ints * sizeof(int)),
         MPI_BYTE, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        MPI_Request r1, r2;
        //MPI_Irecv(edges, (uint64_t)nnz * sizeof(edge_t<T>), MPI_BYTE, src_rank,
        //          0, MPI_COMM_WORLD, &r1);
         MPI_Irecv(value, (uint64_t) (m * n * sizeof(T)), MPI_BYTE,
         src_rank, 0, MPI_COMM_WORLD, &r1);
         MPI_Irecv(bit_vector, (uint64_t) (num_ints * sizeof(int)),
         MPI_BYTE, src_rank, 0, MPI_COMM_WORLD, &r2);
        (*reqs).push_back(r1);
        (*reqs).push_back(r2);
      }
//#pragma omp parallel for
/*
      for (int nz = 0; nz < nnz; nz++) {
        int i = edges[nz].src - 1;
        int j = edges[nz].dst - 1;
        value[i + j * m] = edges[nz].val;
        set_bitvector(i + j * m, bit_vector); 
      }
      delete[] edges;
      */
    }
  }
};

#endif  // SRC_DENSETILE_H_
