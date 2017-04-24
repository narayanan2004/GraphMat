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


#pragma once 

#include <algorithm>

#include "GMDP/matrices/SpMat.h"
#include "GMDP/vectors/SpVec.h"
#include "GMDP/singlenode/applyedges.h"

/* function pointer:
 * apply_fn(edge_val, dst_vertex, src_vertex)
 */

template <template <typename> class SpTile, template<typename> class SpSegment, typename Ta,
          typename Tvp>
void ApplyEdges(SpMat<SpTile<Ta> > * grida, 
                 SpVec<SpSegment<Tvp> >* vecvp,
                 void (*fp)(Ta*, Tvp, Tvp, void*), void* vsp=NULL) {
  int start_m = 0;
  int start_n = 0;
  int end_m = grida->ntiles_y;
  int end_n = grida->ntiles_x;

  // Build list of row/column partners
  std::vector<std::set<int> > row_ranks;
  std::vector<std::set<int> > col_ranks;
  std::vector<std::set<int> > rowcol_ranks;
  get_row_ranks(grida, &row_ranks, &col_ranks);
  for(int j = start_n ; j < end_n ; j++)
  {
    std::set<int> newranks;
    for(auto it = row_ranks[j].begin() ; it != row_ranks[j].end() ; it++)
    {
      newranks.insert(*it);
    }
    for(auto it = col_ranks[j].begin() ; it != col_ranks[j].end() ; it++)
    {
      newranks.insert(*it);
    }
    rowcol_ranks.push_back(newranks);
  }
  std::vector<MPI_Request> requests;

  int global_nrank = get_global_nrank();
  int global_myrank = get_global_myrank();

  double tmp_time = MPI_Wtime();

  // Broadcast input to all nodes in column
  for (int j = start_n; j < end_n; j++) {
    for (std::set<int>::iterator it = rowcol_ranks[j].begin();
         it != rowcol_ranks[j].end(); it++) {
      int dst_rank = *it;
      if (global_myrank == vecvp->nodeIds[j] && global_myrank != dst_rank) {
        vecvp->segments[j]->compress();
      }
    }
  }

  for (int j = start_n; j < end_n; j++) {
    for (std::set<int>::iterator it = rowcol_ranks[j].begin();
         it != rowcol_ranks[j].end(); it++) {
      int dst_rank = *it;
      if (global_myrank == vecvp->nodeIds[j] && global_myrank != dst_rank) {
        vecvp->segments[j]->send_nnz(global_myrank, dst_rank, &requests);
      }
      if (global_myrank == dst_rank && global_myrank != vecvp->nodeIds[j]) {
        vecvp->segments[j]
            ->recv_nnz(global_myrank, vecvp->nodeIds[j], &requests);
      }
    }
  }

  for (int j = start_n; j < end_n; j++) {
    for (std::set<int>::iterator it = rowcol_ranks[j].begin();
         it != rowcol_ranks[j].end(); it++) {
      int dst_rank = *it;
      if (global_myrank == vecvp->nodeIds[j] && global_myrank != dst_rank) {
        vecvp->segments[j]->send_segment(global_myrank, dst_rank, &requests);
      }
      if (global_myrank == dst_rank && global_myrank != vecvp->nodeIds[j]) {
        vecvp->segments[j]
            ->recv_segment(global_myrank, vecvp->nodeIds[j], &requests);
      }
    }
  }


  // Wait_all
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  requests.clear();

  for (int j = start_n; j < end_n; j++) {
    for (std::set<int>::iterator it = rowcol_ranks[j].begin();
         it != rowcol_ranks[j].end(); it++) {
      int dst_rank = *it;
      if (global_myrank != vecvp->nodeIds[j] && global_myrank == dst_rank) {
        vecvp->segments[j]->decompress();
      }
    }
  }

  //spmspv_send_time += MPI_Wtime() - tmp_time;
  tmp_time = MPI_Wtime();

  // Multiply all tiles
  for (int i = start_m; i < end_m; i++) {
    for (int j = start_n; j < end_n; j++) {
      if (global_myrank == grida->nodeIds[i + j * grida->ntiles_y]) {
        apply_edges(grida->tiles[i][j], vecvp->segments[j], vecvp->segments[i], 
                     fp, vsp);
      }
    }
  }

  //spmspv_mult_time += MPI_Wtime() - tmp_time;
  tmp_time = MPI_Wtime();

  // Free input vectors allocated during computation
  for (int j = start_n; j < end_n; j++) {
    if (global_myrank != vecvp->nodeIds[j]) {
      vecvp->segments[j]->set_uninitialized();
    }
  }

  //spmspv_reduce_send_time += MPI_Wtime() - tmp_time;
  tmp_time = MPI_Wtime();

  //spmspv_reduce_time += MPI_Wtime() - tmp_time;
}

