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


#ifndef SRC_CSSEGMENT_H_
#define SRC_CSSEGMENT_H_

#include <string>
#include "src/edgelist.h"
#include "src/bitvector.h"

template <typename T>
class CSSegment {
 public:
  std::string name;
  int capacity;
  int num_ints;
  T* value;
  int * bit_vector;
  bool empty_flag;
  int nnz;

  CSSegment() {
    capacity = 0;
    num_ints = 0;
    empty_flag = true;
    nnz = 0;
  }

  CSSegment(int n)
      : capacity(n) {
    num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
    empty_flag = true;
    nnz = 0;
  }

  CSSegment(edge_t<T>* edges, int _m, int _nnz, int row_start)
      : name("TEMP"), capacity(_m) {
    num_ints = (capacity + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
    alloc();
    nnz = 0;
    for (uint64_t i = 0; i < (uint64_t)_nnz; i++) {
      int src = edges[i].src - row_start - 1;
      set_bitvector(src, bit_vector);
      value[src] = edges[i].val;
      nnz++;
    }
  }

  ~CSSegment() {}

  void clear() {
    if (!isEmpty()) {
      _mm_free(value);
      empty_flag = true;
      nnz = 0;
    }
  }

  void alloc() {
    value = reinterpret_cast<T*>(_mm_malloc(capacity * sizeof(T) + num_ints*sizeof(int), 64));
    bit_vector = reinterpret_cast<int*>( value + capacity);
    memset(bit_vector, 0, num_ints* sizeof(int));
    empty_flag = false;
    nnz = 0;
  }

  bool isEmpty() const { return empty_flag; }

  void set(int idx, T val) {
    if (isEmpty()) {
      alloc();
    }
    value[idx - 1] = val;
    set_bitvector(idx-1, bit_vector);
    nnz++;
  }

  T get(const int idx) const {
    assert(!isEmpty());
    return value[idx - 1];
  }

  void send_tile(int myrank, int dst_rank, int output_rank, std::vector<MPI_Request>* requests) {
    MPI_Send(&capacity, 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&num_ints, 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
    MPI_Send(&nnz, 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
    MPI_Request r1, r2;
    MPI_Isend(value, capacity * sizeof(T) + num_ints * sizeof(int), MPI_BYTE, dst_rank, 0,
             MPI_COMM_WORLD, &r1);
    requests->push_back(r1);
  }

  void recv_tile(int myrank, int src_rank, int output_rank,
                 std::vector<MPI_Request>* requests) {

    MPI_Recv(&capacity, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(&num_ints, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    if (isEmpty()) {
      alloc();
    }
    MPI_Recv(&nnz, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Request r1, r2;
    MPI_Irecv(value, capacity * sizeof(T) + num_ints * sizeof(int), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r1);
    requests->push_back(r1);
  }
};
#endif  // SRC_CSSEGMENT_H_
