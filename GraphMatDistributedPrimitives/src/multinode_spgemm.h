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


#ifndef SRC_MULTINODE_SPGEMM_H_
#define SRC_MULTINODE_SPGEMM_H_

#include <vector>
#include <set>

#include "src/SpMat.h"
#include "src/TileOps.h"

template <template <typename> class SpTile, typename T>
void get_row_ranks(const SpMat<SpTile<T> >& mat,
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

template <template <typename> class ATile, template <typename> class BTile,
          template <typename> class CTile, typename Ta, typename Tb,
          typename Tc>
void SpGEMM_tile_outerproduct(const SpMat<ATile<Ta> >& grida,
                              const SpMat<BTile<Tb> >& gridb,
                              SpMat<CTile<Tc> >* gridc, int start_m,
                              int start_n, int start_k, int end_m, int end_n,
                              int end_k, void (*mul_fp)(Ta, Tb, Tc*, void*),
                              void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
  int output_rank = 0;
  double start, end;

  double* compute_tstamp = new double[(end_k - start_k) * 4];

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  // Calculate row_ranks and col_ranks
  std::vector<std::set<int> > c_row_ranks;
  std::vector<std::set<int> > c_col_ranks;
  get_row_ranks((*gridc), &c_row_ranks, &c_col_ranks);

  bool block = false;
  bool comm_barrier = false;

  // For each row/column, broadcast across columns/rows, then multiply and add
  double total_comm_time = 0.0;
  double total_comp_time = 0.0;
  double total_time = 0.0;
  uint64_t total_recv_bytes = 0;
  for (int k = start_k; k < end_k; k++) {
    compute_tstamp[0 + (k - start_k) * 4] = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (global_myrank == output_rank) {
      std::cout << "Starting row/col: " << k << std::endl;
    }

    std::vector<MPI_Request> requests;
    // For each tile in A col, send/recv
    for (int i = start_m; i < end_m; i++) {
      std::set<int> row_ranks = c_row_ranks[i];
      for (std::set<int>::iterator it = row_ranks.begin();
           it != row_ranks.end(); it++) {
        int dst_rank = *it;
        if (global_myrank == grida.nodeIds[i + k * grida.ntiles_y] &&
            global_myrank != dst_rank) {
          grida.tiles[i][k]
              .send_tile_metadata(global_myrank, dst_rank, output_rank);
          grida.tiles[i][k]
              .send_tile(global_myrank, dst_rank, output_rank, block, &requests);
        }
        if (global_myrank == dst_rank &&
            global_myrank != grida.nodeIds[i + k * grida.ntiles_y]) {
          grida.tiles[i][k].recv_tile_metadata(
              global_myrank, grida.nodeIds[i + k * grida.ntiles_y],
              output_rank);
          grida.tiles[i][k].recv_tile(global_myrank,
                                      grida.nodeIds[i + k * grida.ntiles_y],
                                      output_rank, block, &requests);
          total_recv_bytes +=
              (grida.tiles[i][k].nnz) * (sizeof(Ta) + sizeof(int)) +
              grida.tiles[i][k].m * sizeof(int);
        }
      }
    }

    // For each tile in B row, send/recv
    for (int j = start_n; j < end_n; j++) {
      std::set<int> col_ranks = c_col_ranks[j];
      for (std::set<int>::iterator it = col_ranks.begin();
           it != col_ranks.end(); it++) {
        int dst_rank = *it;
        if (global_myrank == gridb.nodeIds[k + j * gridb.ntiles_y] &&
            global_myrank != dst_rank) {
          gridb.tiles[k][j]
              .send_tile_metadata(global_myrank, dst_rank, output_rank);
          gridb.tiles[k][j]
              .send_tile(global_myrank, dst_rank, output_rank, block, &requests);
        }
        if (global_myrank == dst_rank &&
            global_myrank != gridb.nodeIds[k + j * gridb.ntiles_y]) {
          gridb.tiles[k][j].recv_tile_metadata(
              global_myrank, gridb.nodeIds[k + j * gridb.ntiles_y],
              output_rank);
          gridb.tiles[k][j].recv_tile(global_myrank,
                                      gridb.nodeIds[k + j * gridb.ntiles_y],
                                      output_rank, block, &requests);
          total_recv_bytes +=
              (gridb.tiles[k][j].nnz) * (sizeof(Tb) + sizeof(int)) +
              gridb.tiles[k][j].m * sizeof(int);
        }
      }
    }

    compute_tstamp[1 + (k - start_k) * 4] = MPI_Wtime();
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
    compute_tstamp[2 + (k - start_k) * 4] = MPI_Wtime();

    for (int i = start_m; i < end_m; i++) {
      for (int j = start_n; j < end_n; j++) {
        if (global_myrank == gridc->nodeIds[i + j * gridc->ntiles_y]) {
          ATile<Ta> tilea = grida.tiles[i][k];
          BTile<Tb> tileb = gridb.tiles[k][j];
          mult_tile(tilea, tileb, &(gridc->tiles[i][j]), global_myrank,
                    output_rank, mul_fp, add_fp, vsp);
        }
      }
    }

    // For each tile in A col, free
    for (int i = start_m; i < end_m; i++) {
      if (global_myrank != grida.nodeIds[i + k * grida.ntiles_y]) {
        grida.tiles[i][k].clear();
      }
    }

    for (int j = start_n; j < end_n; j++) {
      if (global_myrank != gridb.nodeIds[k + j * gridb.ntiles_y]) {
        gridb.tiles[k][j].clear();
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    compute_tstamp[3 + (k - start_k) * 4] = MPI_Wtime();
    double comp_end = MPI_Wtime();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  // Instrumentation
  double* starts = new double[global_nrank];
  double* ends = new double[global_nrank];
  double** compute_tstamps = new double* [global_nrank];
  for (int i = 0; i < global_nrank; i++) {
    compute_tstamps[i] = new double[(end_k - start_k) * 4];
  }
  starts[global_myrank] = start;
  ends[global_myrank] = end;
  memcpy(compute_tstamps[global_myrank], compute_tstamp,
         (end_k - start_k) * 4 * sizeof(double));

  uint64_t* recv_bytes = new uint64_t[global_nrank];
  uint64_t* mul_counts = new uint64_t[global_nrank];
  uint64_t* add_counts = new uint64_t[global_nrank];
  recv_bytes[global_myrank] = total_recv_bytes;
  mul_counts[global_myrank] = mul_flops;
  add_counts[global_myrank] = add_flops;
  if (global_myrank != output_rank) {
    MPI_Send(starts + global_myrank, 1, MPI_DOUBLE, output_rank, 0,
             MPI_COMM_WORLD);
    MPI_Send(ends + global_myrank, 1, MPI_DOUBLE, output_rank, 0,
             MPI_COMM_WORLD);
    MPI_Send(compute_tstamps[global_myrank], (end_k - start_k) * 4, MPI_DOUBLE,
             output_rank, 0, MPI_COMM_WORLD);

    MPI_Send(recv_bytes + global_myrank, 1, MPI_UNSIGNED_LONG, output_rank, 0,
             MPI_COMM_WORLD);
    MPI_Send(add_counts + global_myrank, 1, MPI_UNSIGNED_LONG, output_rank, 0,
             MPI_COMM_WORLD);
    MPI_Send(mul_counts + global_myrank, 1, MPI_UNSIGNED_LONG, output_rank, 0,
             MPI_COMM_WORLD);
  }
  if (global_myrank == output_rank) {
    for (int i = 0; i < global_nrank; i++) {
      if (i != output_rank) {
        MPI_Recv(starts + i, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(ends + i, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(compute_tstamps[i], (end_k - start_k) * 4, MPI_DOUBLE, i, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Recv(recv_bytes + i, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(add_counts + i, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(mul_counts + i, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
    }
  }

  if (global_myrank == output_rank) {
    std::cout << " === SpGEMM Summary === " << std::endl;
    for (int i = 0; i < global_nrank; i++) {
      double comm_time = 0;
      double comp_time = 0;
      for (int j = 0; j < (end_k - start_k); j++) {
        comm_time +=
            (compute_tstamps[i][1 + j * 4] - compute_tstamps[i][0 + j * 4]);
        comp_time +=
            (compute_tstamps[i][3 + j * 4] - compute_tstamps[i][2 + j * 4]);
      }
      std::cout << "Rank: " << i << std::endl;
      std::cout << "Start: " << starts[i] << std::endl;
      std::cout << "End: " << ends[i] << std::endl;
      std::cout << "Total time: " << ends[i] - starts[i] << std::endl;
      std::cout << "Comm time: " << comm_time << std::endl;
      std::cout << "Comp time: " << comp_time << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

template <template <typename> class ATile, template <typename> class BTile,
          template <typename> class CTile, typename Ta, typename Tb,
          typename Tc>
void SpGEMM_tile_innerproduct(const SpMat<ATile<Ta> >& grida,
                              const SpMat<BTile<Tb> >& gridb,
                              SpMat<CTile<Tc> >* gridc, int start_m,
                              int start_n, int start_k, int end_m, int end_n,
                              int end_k, Tc (*mul_fp)(Ta, Tb),
                              Tc (*add_fp)(Tc, Tc)) {
  int output_rank = 0;

  double multime = 0.0;
  double addtime = 0.0;
  double istart, iend;

  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  // Get row ranks and column ranks
  std::vector<std::set<int> > a_row_ranks;
  std::vector<std::set<int> > a_col_ranks;
  get_row_ranks(grida, &a_row_ranks, &a_col_ranks);

  bool block = true;

  double total_comm_time = 0.0;
  double total_comp_time = 0.0;

  // For each row/column, broadcast across columns/rows, then multiply and add
  for (int j = start_n; j < end_n; j++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (global_myrank == output_rank) {
      std::cout << "Starting B col: " << j << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double comm_start = MPI_Wtime();
    std::vector<MPI_Request> requests;

    // For each tile in B col j, send/recv to all tiles in A col j
    for (int i = start_k; i < end_k; i++) {
      std::set<int> col_ranks = a_col_ranks[i];
      for (std::set<int>::iterator it = col_ranks.begin();
           it != col_ranks.end(); it++) {
        int dst_rank = *it;
        if (global_myrank == gridb.nodeIds[i + j * gridb.ntiles_y] &&
            global_myrank != dst_rank) {
          gridb.tiles[i][j]
              .send_tile_metadata(global_myrank, dst_rank, output_rank);
          gridb.tiles[i][j]
              .send_tile(global_myrank, dst_rank, output_rank, block, &requests);
        }
        if (global_myrank == dst_rank &&
            global_myrank != gridb.nodeIds[i + j * gridb.ntiles_y]) {
          gridb.tiles[i][j].recv_tile_metadata(
              global_myrank, gridb.nodeIds[i + j * gridb.ntiles_y],
              output_rank);
          gridb.tiles[i][j].recv_tile(global_myrank,
                                      gridb.nodeIds[i + j * gridb.ntiles_y],
                                      output_rank, block, &requests);
        }
      }
    }

    double comm_end = MPI_Wtime();
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
    MPI_Barrier(MPI_COMM_WORLD);
    double comp_start = MPI_Wtime();

    // For each tile in C col j, multiply/add
    for (int i = start_m; i < end_m; i++) {
      // For each tile in row i of A and col j of B
      for (int k = start_k; k < end_k; k++) {
        if (global_myrank == grida.nodeIds[i + k * grida.ntiles_y]) {
          mult_tile(grida.tiles[i][k], gridb.tiles[k][j], &(gridc->tiles[i][j]),
                    global_myrank, output_rank, mul_fp, add_fp);
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double comp_end = MPI_Wtime();
    double comm_start2 = MPI_Wtime();

    // For each tile in C col j, Reduce C tiles across all nodes in A row i
    for (int i = start_m; i < end_m; i++) {
      std::set<int> row_ranks = a_row_ranks[i];
      std::vector<int> row_ranks_vec;
      for (std::set<int>::iterator it = row_ranks.begin();
           it != row_ranks.end(); it++) {
        row_ranks_vec.push_back(*it);
      }
      for (int stride = 1; stride < row_ranks.size(); stride *= 2) {
        for (int idx = 0; idx < row_ranks.size(); idx += stride) {
          if (idx + stride >= row_ranks.size()) break;
          int src_rank = row_ranks_vec[idx + stride];
          int dst_rank = row_ranks_vec[idx];
          if (global_myrank == src_rank) {
            assert(global_myrank != dst_rank);
            gridc->tiles[i][j]
                .send_tile_metadata(global_myrank, dst_rank, output_rank);
            gridc->tiles[i][j]
                .send_tile(global_myrank, dst_rank, output_rank, true, &requests);
            gridc->tiles[i][j].clear();
          }
          if (global_myrank == dst_rank) {
            assert(global_myrank != src_rank);
            CTile<Tc> tiletc;
            tiletc.recv_tile_metadata(global_myrank, src_rank, output_rank);
            tiletc.recv_tile(global_myrank, src_rank, output_rank, true,
                             &requests);
            union_tile(tiletc, &(gridc->tiles[i][j]), global_myrank,
                       output_rank, add_fp);
          }
        }
      }

      // Result in row_ranks_vec[0]
      int src_rank = row_ranks_vec[0];
      int dst_rank = gridc->nodeIds[i + j * gridc->ntiles_y];
      if (global_myrank == src_rank && src_rank != dst_rank) {
        gridc->tiles[i][j]
            .send_tile_metadata(global_myrank, dst_rank, output_rank);
        gridc->tiles[i][j]
            .send_tile(global_myrank, dst_rank, output_rank, true, &requests);
        gridc->tiles[i][j].clear();
      }
      if (global_myrank == dst_rank && src_rank != dst_rank) {
        gridc->tiles[i][j].clear();
        gridc->tiles[i][j]
            .recv_tile_metadata(global_myrank, src_rank, output_rank);
        gridc->tiles[i][j]
            .recv_tile(global_myrank, src_rank, output_rank, true, &requests);
      }
    }

    // For each tile in B col j, free
    for (int i = start_k; i < end_k; i++) {
      if (global_myrank != gridb.nodeIds[i + j * gridb.ntiles_y]) {
        gridb.tiles[i][j].clear();
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double comm_end2 = MPI_Wtime();
    total_comm_time += (comm_end - comm_start) + (comm_end2 - comm_start2);
    total_comp_time += comp_end - comp_start;
  }

  double end = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  double end_barrier = MPI_Wtime();

  if (global_myrank == output_rank) {
    std::cout << "TOTAL TIME: " << end_barrier - start << std::endl;
  }
  if (global_myrank == output_rank) {
    std::cout << "RANK: " << global_myrank
              << " communication time: " << total_comm_time
              << "\t compute time: " << total_comp_time << std::endl;
  }
}

#endif  // SRC_MULTINODE_SPGEMM_H_
