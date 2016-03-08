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


#ifndef SRC_MULTINODE_REDUCE_H_
#define SRC_MULTINODE_REDUCE_H_

#include "src/SpMat.h"
#include "src/TileOps.h"

template <template <typename> class SpTile, typename T>
void get_row_ranks_reduce(const SpMat<SpTile<T> >& mat,
                          std::vector<std::set<int> >* row_ranks_out) {
  for (int i = 0; i < mat.ntiles_y; i++) {
    // Create set of row nodeIDs
    std::set<int> row_ranks;
    for (int j = 0; j < mat.ntiles_x; j++) {
      row_ranks.insert(mat.nodeIds[i + j * mat.ntiles_y]);
    }
    row_ranks_out->push_back(row_ranks);
  }
}

template <template <typename> class SpTile, template<typename> class SpSegment, typename T>
void Reduce_tile(const SpMat<SpTile<T> >& mat, SpVec<SpSegment<T> > * vecy, int start_x, int start_y,
                 int end_x, int end_y, void (*op_fp)(T, T, T*, void*), void* vsp) {
  int output_rank = 0;

  // Build list of row/column partners
  std::vector<std::set<int> > row_ranks;
  get_row_ranks_reduce(mat, &row_ranks);

  // Reduce all tiles
  for (int i = start_y; i < end_y; i++) {
    for (int j = start_x; j < end_x; j++) {
      if (global_myrank == mat.nodeIds[i + j * mat.ntiles_y]) {
        reduce_tile(mat.tiles[i][j], &(vecy->segments[i]),
                     op_fp, vsp);
      }
    }
  }
  std::vector<MPI_Request> requests;

  // Reduce across rows
  for (int i = start_y; i < end_y; i++) {
    std::vector<DenseSegment<T> > row_segments;
    for (std::set<int>::iterator it = row_ranks[i].begin();
         it != row_ranks[i].end(); it++) {
      int src_rank = *it;
      if (global_myrank == vecy->nodeIds[i] && global_myrank != src_rank) {
        vecy->segments[i].recv_tile_metadata(global_myrank, src_rank, output_rank, &requests);
      }
      if (global_myrank != vecy->nodeIds[i] && global_myrank == src_rank) {
        vecy->segments[i]
            .send_tile_metadata(global_myrank, vecy->nodeIds[i], output_rank, &requests);
      }
    }
  }

  // Reduce across rows
  for (int i = start_y; i < end_y; i++) {
    std::vector<DenseSegment<T> > row_segments;
    for (std::set<int>::iterator it = row_ranks[i].begin();
         it != row_ranks[i].end(); it++) {
      int src_rank = *it;
      if (global_myrank == vecy->nodeIds[i] && global_myrank != src_rank) {
        vecy->segments[i].recv_tile(global_myrank, src_rank, output_rank, &requests);
      }
      if (global_myrank != vecy->nodeIds[i] && global_myrank == src_rank) {
        vecy->segments[i]
            .send_tile(global_myrank, vecy->nodeIds[i], output_rank, &requests);
      }
    }
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  requests.clear();

  // Free any output vectors allocated during computation
  for (int i = start_y; i < end_y; i++) {
    if (global_myrank != vecy->nodeIds[i]) {
      vecy->segments[i].set_uninitialized();
    }
  }

  // Sum received tiles
  for (int i = start_y; i < end_y; i++) {
    if (global_myrank == vecy->nodeIds[i]) {
      union_compress_segment(&(vecy->segments[i]), op_fp, vsp);
      vecy->segments[i].set_uninitialized_received();
    }
  }
}


template <template <typename> class SpTile, typename T>
void Reduce_tile(const SpMat<SpTile<T> >& mat, T* res, int start_x, int start_y,
                 int end_x, int end_y, void (*op_fp)(T, T, T*, void*), void* vsp) {
  bool res_set = false;

  // Count triangles
  for (int i = start_y; i < end_y; i++) {
    for (int j = start_x; j < end_x; j++) {
      if (mat.nodeIds[i + j * mat.ntiles_y] == global_myrank) {
        SpTile<T> tilea = mat.tiles[i][j];
        reduce_tile(tilea, res, &res_set, op_fp, vsp);
      }
    }
  }
  if (!res_set) {
    *res = 0;
  }

  // Reduce across nodes
  T* all_res = new T[global_nrank];
  all_res[global_myrank] = *res;
  MPI_Status status;
  for (int i = 1; i < global_nrank; i++) {
    if (global_myrank == i) {
      MPI_Send(all_res + i, sizeof(T), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
  }
  for (int i = 1; i < global_nrank; i++) {
    if (global_myrank == 0) {
      MPI_Recv(all_res + i, sizeof(T), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      T res_tmp = *res;
      op_fp(res_tmp, all_res[i], res, vsp);
    }
  }

  // Broadcast to all nodes
  MPI_Bcast(res, sizeof(T), MPI_CHAR, 0, MPI_COMM_WORLD);
}

template <template<typename> class SpSegment, typename T>
void Reduce_tile(const SpVec<SpSegment<T> >& vec, T* res, int start, int end,
                 void (*op_fp)(T, T, T*, void*), void* vsp) {
  bool res_set = false;

  for (int i = start; i < end; i++) {
    if (vec.nodeIds[i] == global_myrank) {
      DenseSegment<T> segment = vec.segments[i];
      reduce_segment(segment, res, &res_set, op_fp, vsp);
    }
  }

  // Reduce across nodes
  T* all_res = new T[global_nrank];
  all_res[global_myrank] = *res;
  MPI_Status status;
  for (int i = 1; i < global_nrank; i++) {
    if (global_myrank == i) {
      MPI_Send(all_res + i, sizeof(T), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
  }
  for (int i = 1; i < global_nrank; i++) {
    if (global_myrank == 0) {
      MPI_Recv(all_res + i, sizeof(T), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      T tmp_res = *res;
      op_fp(tmp_res, all_res[i], res, vsp);
    }
  }

  // Broadcast to all nodes
  MPI_Bcast(res, sizeof(T), MPI_CHAR, 0, MPI_COMM_WORLD);
}


template <template<typename> class SpSegment, typename T, typename VT>
void MapReduce_tile(SpVec<SpSegment<VT> > * vec, T* res, int start, int end,
                 void (*op_map)(VT*, T*, void*), void (*op_fp)(T, T, T*, void*), void* vsp) {
  bool res_set = false;

  for (int i = start; i < end; i++) {
    if (vec->nodeIds[i] == global_myrank) {
      mapreduce_segment(&(vec->segments[i]), res, &res_set, op_map, op_fp, vsp);
    }
  }

  // Reduce across nodes
  T* all_res = new T[global_nrank];
  all_res[global_myrank] = *res;
  MPI_Status status;
  for (int i = 1; i < global_nrank; i++) {
    if (global_myrank == i) {
      MPI_Send(all_res + i, sizeof(T), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
  }
  for (int i = 1; i < global_nrank; i++) {
    if (global_myrank == 0) {
      MPI_Recv(all_res + i, sizeof(T), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      T tmp_res = *res;
      op_fp(tmp_res, all_res[i], res, vsp);
    }
  }

  // Broadcast to all nodes
  MPI_Bcast(res, sizeof(T), MPI_CHAR, 0, MPI_COMM_WORLD);
}



#endif  // SRC_MULTINODE_REDUCE_H_
