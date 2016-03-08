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


#ifndef SRC_MULTINODE_FSPMSPV_H_
#define SRC_MULTINODE_FSPMSPV_H_

#include "src/SpMat.h"
#include "src/TileOps.h"
#include "src/SpVec.h"

template <template <typename> class SpTile, typename T>
void get_row_ranks_fspmspv(const SpMat<SpTile<T> >& mat,
                          std::vector<std::set<int> >* row_ranks_out,
                          std::vector<std::set<int> >* col_ranks_out) {
  for (int i = 0; i < mat.ntiles_y; i++) {
    // Create set of row nodeIDs
    std::set<int> row_ranks;
    for (int j = 0; j < mat.ntiles_x; j++) {
      row_ranks.insert(mat.nodeIds[i + j * mat.ntiles_y]);
    }
    row_ranks_out->push_back(row_ranks);
  }

  for (int j = 0; j < mat.ntiles_x; j++) {
    // Create set of col nodeIDs
    std::set<int> col_ranks;
    for (int i = 0; i < mat.ntiles_y; i++) {
      col_ranks.insert(mat.nodeIds[i + j * mat.ntiles_y]);
    }
    col_ranks_out->push_back(col_ranks);
  }
}

template <template <typename> class SpTile, template<typename> class SpSegment, typename Ta, typename Tx,
          typename Ty, typename Tm>
void FSpMSpV_tile(const SpMat<SpTile<Ta> >& grida, const SpVec<SpSegment<Tx> >& vecx, const SpVec<SpSegment<Tm> >& vecm,
                 SpVec<SpSegment<Ty> >* vecy, int start_m, int start_n, int end_m,
                 int end_n, void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp=NULL) {
  int output_rank = 0;

  // Build list of row/column partners
  std::vector<std::set<int> > row_ranks;
  std::vector<std::set<int> > col_ranks;
  get_row_ranks(grida, &row_ranks, &col_ranks);
  std::vector<MPI_Request> requests;

  double tmp_time = MPI_Wtime();

  // Broadcast input to all nodes in column
  for (int j = start_n; j < end_n; j++) {
    for (std::set<int>::iterator it = col_ranks[j].begin();
         it != col_ranks[j].end(); it++) {
      int dst_rank = *it;
      if (global_myrank == vecx.nodeIds[j] && global_myrank != dst_rank) {
        vecx.segments[j].compress();
      }
      if (global_myrank == vecm.nodeIds[j] && global_myrank != dst_rank) {
        vecm.segments[j].compress();
      }
    }
  }


  // Broadcast input to all nodes in column
  for (int j = start_n; j < end_n; j++) {
    for (std::set<int>::iterator it = col_ranks[j].begin();
         it != col_ranks[j].end(); it++) {
      int dst_rank = *it;
      if (global_myrank == vecx.nodeIds[j] && global_myrank != dst_rank) {
        vecx.segments[j].send_tile_metadata(global_myrank, dst_rank, output_rank, &requests);
      }
      if (global_myrank == vecm.nodeIds[j] && global_myrank != dst_rank) {
        vecm.segments[j].send_tile_metadata(global_myrank, dst_rank, output_rank, &requests);
      }
      if (global_myrank == dst_rank && global_myrank != vecx.nodeIds[j]) {
        vecx.segments[j]
            .recv_tile_metadata(global_myrank, vecx.nodeIds[j], output_rank, &requests);
      }
      if (global_myrank == dst_rank && global_myrank != vecm.nodeIds[j]) {
        vecm.segments[j]
            .recv_tile_metadata(global_myrank, vecm.nodeIds[j], output_rank, &requests);
      }
    }
  }

  // Broadcast input to all nodes in column
  for (int j = start_n; j < end_n; j++) {
    for (std::set<int>::iterator it = col_ranks[j].begin();
         it != col_ranks[j].end(); it++) {
      int dst_rank = *it;
      if (global_myrank == vecx.nodeIds[j] && global_myrank != dst_rank) {
        vecx.segments[j].send_tile_compressed(global_myrank, dst_rank, output_rank, &requests);
      }
      if (global_myrank == vecm.nodeIds[j] && global_myrank != dst_rank) {
        vecm.segments[j].send_tile_compressed(global_myrank, dst_rank, output_rank, &requests);
      }
      if (global_myrank == dst_rank && global_myrank != vecx.nodeIds[j]) {
        vecx.segments[j]
            .recv_tile_compressed(global_myrank, vecx.nodeIds[j], output_rank, &requests);
      }
      if (global_myrank == dst_rank && global_myrank != vecm.nodeIds[j]) {
        vecm.segments[j]
            .recv_tile_compressed(global_myrank, vecm.nodeIds[j], output_rank, &requests);
      }
    }
  }

  // Wait_all
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  requests.clear();

  for (int j = start_n; j < end_n; j++) {
    for (std::set<int>::iterator it = col_ranks[j].begin();
         it != col_ranks[j].end(); it++) {
      int dst_rank = *it;
      if (global_myrank == dst_rank && global_myrank != vecx.nodeIds[j]) {
        vecx.segments[j]
            .decompress();
      }
      if (global_myrank == dst_rank && global_myrank != vecm.nodeIds[j]) {
        vecm.segments[j]
            .decompress();
      }
    }
  }

  spmspv_send_time += MPI_Wtime() - tmp_time;
  tmp_time = MPI_Wtime();

  // Multiply all tiles
  for (int i = start_m; i < end_m; i++) {
    for (int j = start_n; j < end_n; j++) {
      if (global_myrank == grida.nodeIds[i + j * grida.ntiles_y]) {
        fmult_segment(grida.tiles[i][j], vecx.segments[j], vecm.segments[j], &(vecy->segments[i]),
                     output_rank, mul_fp, add_fp, vsp);
      }
    }
  }

  spmspv_mult_time += MPI_Wtime() - tmp_time;
  tmp_time = MPI_Wtime();

  // Free input vectors allocated during computation
  for (int j = start_n; j < end_n; j++) {
    if (global_myrank != vecx.nodeIds[j]) {
      vecx.segments[j].set_uninitialized();
    }
    if (global_myrank != vecm.nodeIds[j]) {
      vecm.segments[j].set_uninitialized();
    }
  }

  // Reduce across rows
  for (int i = start_m; i < end_m; i++) {
    for (std::set<int>::iterator it = row_ranks[i].begin();
         it != row_ranks[i].end(); it++) {
      int src_rank = *it;
      if (global_myrank != vecy->nodeIds[i] && global_myrank == src_rank) {
        vecy->segments[i]
            .compress();
      }
    }
  }


  // Reduce across rows
  for (int i = start_m; i < end_m; i++) {
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
  for (int i = start_m; i < end_m; i++) {
    for (std::set<int>::iterator it = row_ranks[i].begin();
         it != row_ranks[i].end(); it++) {
      int src_rank = *it;
      if (global_myrank == vecy->nodeIds[i] && global_myrank != src_rank) {
	vecy->segments[i].recv_tile_compressed(global_myrank, src_rank, output_rank, &requests);
      }
      if (global_myrank != vecy->nodeIds[i] && global_myrank == src_rank) {
        vecy->segments[i]
            .send_tile_compressed(global_myrank, vecy->nodeIds[i], output_rank, &requests);
      }
    }
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  requests.clear();

  spmspv_reduce_send_time += MPI_Wtime() - tmp_time;
  tmp_time = MPI_Wtime();

  // Free any output vectors allocated during computation
  for (int i = start_m; i < end_m; i++) {
    if (global_myrank != vecy->nodeIds[i]) {
      vecy->segments[i].set_uninitialized();
    }
  }

  // Sum received tiles
  for (int i = start_m; i < end_m; i++) {
    if (global_myrank == vecy->nodeIds[i]) {
      union_compress_segment(&(vecy->segments[i]), add_fp, vsp);
      vecy->segments[i].set_uninitialized_received();
    }
  }

  spmspv_reduce_time += MPI_Wtime() - tmp_time;
}

#endif  // SRC_MULTINODE_FSPMSPV_H_
