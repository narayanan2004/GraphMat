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
/* Narayanan Sundaram (Intel Corp.), Michael Anderson (Intel Corp.)
 *  * ******************************************************************************/


#ifndef SRC_SPMAT_H_
#define SRC_SPMAT_H_

#include <string>
#include <algorithm>

#include "src/TileOps.h"
#include "binary_search.h"

template <typename T>
bool compare_tile_id(const tedge_t<T>& a, const tedge_t<T>& b) {
  if (a.tile_id < b.tile_id)
    return true;
  return false;
}


template <typename SpTile>
class SpMat {
 public:
  SpTile** tiles;

  int* start_idx;
  int* start_idy;
  int* nodeIds;
  int ntiles_x;
  int ntiles_y;
  int m;
  int n;
  bool empty;
  std::string name;
  int num_tiles_x;
  int num_tiles_y;
  int (*pfn)(int, int, int, int, int);

  SpMat() { empty = true; }

  void set(int _m, int _n, int _ntiles_x, int _ntiles_y, int* _nodeIds,
           int* _start_idx, int* _start_idy) {
/*
if(global_myrank == 0)
{
    for(int i = 0 ; i < _ntiles_y + 1 ; i++)
    {
      std::cout << "_start_idy[" << i << "]: " << _start_idy[i] << std::endl;
    }
    for(int i = 0 ; i < _ntiles_x + 1 ; i++)
    {
      std::cout << "_start_idx[" << i << "]: " << _start_idx[i] << std::endl;
    }
 }
 */
    empty = false;
    // Copy metadata
    start_idx =
        reinterpret_cast<int*>(_mm_malloc((_ntiles_x + 1) * sizeof(int), 64));
    start_idy =
        reinterpret_cast<int*>(_mm_malloc((_ntiles_y + 1) * sizeof(int), 64));
    nodeIds = reinterpret_cast<int*>(
        _mm_malloc((_ntiles_x * _ntiles_y) * sizeof(int), 64));
    memcpy(start_idx, _start_idx, (_ntiles_x + 1) * sizeof(int));
    memcpy(start_idy, _start_idy, (_ntiles_y + 1) * sizeof(int));
    memcpy(nodeIds, _nodeIds, (_ntiles_x * _ntiles_y) * sizeof(int));
    m = _m;
    n = _n;
    ntiles_x = _ntiles_x;
    ntiles_y = _ntiles_y;
    assert(ntiles_x > 0);
    assert(ntiles_y > 0);

    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate space for tiles
    tiles = new SpTile* [ntiles_y];
    for (int i = 0; i < ntiles_y; i++) {
      tiles[i] = new SpTile[ntiles_x];
    }

    // Set metadata
    for (int tile_j = 0; tile_j < ntiles_x; tile_j++) {
      for (int tile_i = 0; tile_i < ntiles_y; tile_i++) {
        int tile_m = start_idy[tile_i + 1] - start_idy[tile_i];
        int tile_n = start_idx[tile_j + 1] - start_idx[tile_j];
        tiles[tile_i][tile_j] = SpTile(tile_m, tile_n);
        std::stringstream ss;
        ss << "Created_" << tile_i << "," << tile_j;
        tiles[tile_i][tile_j].name = ss.str();
      }
    }
  }

  inline int getPartition(const int src, const int dst, int* ival, int* jval) const {
    (*ival) = -1;
    (*jval) = -1;
    for (int i = 0; i < ntiles_y; i++) {
      if ((src > start_idy[i]) && (src <= start_idy[i + 1])) {
        (*ival) = i;
        break;
      }
    }
    for (int j = 0; j < ntiles_x; j++) {
      if ((dst > start_idx[j]) && (dst <= start_idx[j + 1])) {
        (*jval) = j;
        break;
      }
    }
    if ((*ival) == -1 || (*jval) == -1) {
      printf("%d %d == -1\n", src, dst);
      return -1;
    }
    return (*ival) + (*jval) * ntiles_y;
  }

  template <typename T>
  void ingestEdgelist(edgelist_t<T> blob) {
    int nnz_l = blob.nnz;
    edge_t<T>* edge_list = blob.edges;

    int m = blob.m;
    int n = blob.n;

    printf("Rank %d: Before shuffle %d edges\n", global_myrank, blob.nnz);

    edge_t<T> * received_edges;
    unsigned long int new_nnz = 0;
    if(global_nrank == 1)
    {
      new_nnz = nnz_l;
      received_edges = new edge_t<T>[new_nnz];
      memcpy(received_edges, edge_list, new_nnz * sizeof(edge_t<T>));
    }
    else
    {
      tedge_t<T> * tedges = new tedge_t<T>[nnz_l];
      #pragma omp parallel for
      for(unsigned long i = 0 ; i < nnz_l ; i++)
      {
        tedges[i].src = edge_list[i].src;
        tedges[i].dst = edge_list[i].dst;
        tedges[i].val = edge_list[i].val;
        int ival, jval;
        int tile_id = getPartition(edge_list[i].src, edge_list[i].dst, &ival, &jval);
        assert(tile_id != -1);
        tedges[i].tile_id = nodeIds[ival + jval * ntiles_y];
      }
  
      __gnu_parallel::sort(tedges, tedges + nnz_l, compare_tile_id<T>);
  
      int * assignment = new int[nnz_l];
      #pragma omp parallel for
      for(unsigned long i = 0 ; i < nnz_l ; i++)
      {
        edge_list[i].src = tedges[i].src;
        edge_list[i].dst = tedges[i].dst;
        edge_list[i].val = tedges[i].val;
        assignment[i] = tedges[i].tile_id;
      }
  
      delete [] tedges;
  
      unsigned long int * positions = new unsigned long[global_nrank+1];
      unsigned long int * counts = new unsigned long[global_nrank];
      unsigned long int * recv_positions = new unsigned long[global_nrank+1];
      unsigned long int * recv_counts = new unsigned long[global_nrank];
      unsigned long int current_count = 0;
      for(int i = 0 ; i < global_nrank ; i++)
      {
        int point = binary_search_right_border(assignment, i, 0, nnz_l, nnz_l);
        if(point == -1)
        {
          counts[i] = 0;
          positions[i] = current_count;
        }
        else
        {
          counts[i] = (point+1) - current_count;
          positions[i] = current_count;
          current_count = (point+1);
        }
  
        if(global_myrank == 0)
        {
          std::cout << "point: " << point << "\t" << counts[i] << std::endl;
        }
      }
      positions[global_nrank] = nnz_l;
      MPI_Barrier(MPI_COMM_WORLD);
  
      delete [] assignment;
  
      MPI_Request* mpi_req = new MPI_Request[2 * global_nrank];
      MPI_Status* mpi_status = new MPI_Status[2 * global_nrank];
  
      for (int i = 0; i < global_nrank; i++) {
        MPI_Isend(&counts[i], 1, MPI_UNSIGNED_LONG, i, global_myrank, MPI_COMM_WORLD,
                  &mpi_req[i]);
      }
      for (int i = 0; i < global_nrank; i++) {
        MPI_Irecv(&recv_counts[i], 1, MPI_UNSIGNED_LONG, i, i, MPI_COMM_WORLD,
                  &mpi_req[i + global_nrank]);
      }
      MPI_Waitall(2 * global_nrank, mpi_req, mpi_status);
      MPI_Barrier(MPI_COMM_WORLD);
  
  
      recv_positions[0] = 0;
      for(int i = 0 ; i < global_nrank ; i++)
      {
        new_nnz += recv_counts[i];
        recv_positions[i+1] = new_nnz;
      }

      printf("Rank %d: After shuffle %d edges\n", global_myrank, new_nnz);
  
      MPI_Datatype MPI_EDGE_T;
      MPI_Type_contiguous(sizeof(edge_t<T>), MPI_CHAR, &MPI_EDGE_T);
      MPI_Type_commit(&MPI_EDGE_T);
      for (int i = 0; i < global_nrank; i++) {
        MPI_Isend(edge_list + positions[i], counts[i] ,
                  MPI_EDGE_T, i, global_myrank, MPI_COMM_WORLD, &mpi_req[i]);
      }
      received_edges = new edge_t<T>[new_nnz];
  
      for (int i = 0; i < global_nrank; i++) {
        MPI_Irecv(received_edges + recv_positions[i], recv_counts[i] ,
                  MPI_EDGE_T, i, i, MPI_COMM_WORLD, &mpi_req[i+global_nrank]);
      }
  
      MPI_Waitall(2 * global_nrank, mpi_req, mpi_status);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    printf("Rank %d: After shuffle %d edges\n", global_myrank, new_nnz);

    tedge_t<T> * tedges2 = new tedge_t<T>[new_nnz];
    #pragma omp parallel for
    for(unsigned long i = 0 ; i < new_nnz ; i++)
    {
      tedges2[i].src = received_edges[i].src;
      tedges2[i].dst = received_edges[i].dst;
      tedges2[i].val = received_edges[i].val;
      int ival, jval;
      tedges2[i].tile_id = getPartition(received_edges[i].src, received_edges[i].dst, &ival, &jval);
      assert(tedges2[i].tile_id != -1);
    }

    __gnu_parallel::sort(tedges2, tedges2 + new_nnz , compare_tile_id<T>);

    int * assignment2 = new int[new_nnz];
    #pragma omp parallel for
    for(unsigned long i = 0 ; i < new_nnz ; i++)
    {
      received_edges[i].src = tedges2[i].src;
      received_edges[i].dst = tedges2[i].dst;
      received_edges[i].val = tedges2[i].val;
      assignment2[i] = tedges2[i].tile_id;
    }

    delete [] tedges2;

    for (int tile_j = 0; tile_j < ntiles_x; tile_j++) {
      for (int tile_i = 0; tile_i < ntiles_y; tile_i++) {
        if (nodeIds[tile_i + tile_j * ntiles_y] == global_myrank) {
          int tile_m = start_idy[tile_i + 1] - start_idy[tile_i];
          int tile_n = start_idx[tile_j + 1] - start_idx[tile_j];
          int this_tile_id = tile_i + tile_j * ntiles_y;

          // Find left and right
          int start_nz = binary_search_left_border(assignment2, this_tile_id, 0, new_nnz, new_nnz);
          int end_nz = binary_search_right_border(assignment2, this_tile_id, 0, new_nnz, new_nnz) + 1;
          int nnz = 0;
          if((start_nz != -1) && (end_nz != -1))
          {
            nnz = end_nz - start_nz;
          }
          if (nnz <= 0) {
            tiles[tile_i][tile_j] = SpTile(tile_m, tile_n);
            std::stringstream ss;
            ss << "LoadedEmpty_" << tile_i << "," << tile_j;
            tiles[tile_i][tile_j].name = ss.str();
          } else {
            tiles[tile_i][tile_j] =
                SpTile(received_edges + start_nz, tile_m, tile_n, nnz, start_idy[tile_i],
                       start_idx[tile_j]);
            std::stringstream ss;
            ss << "Loaded_" << tile_i << "," << tile_j;
            tiles[tile_i][tile_j].name = ss.str();
          }
        }
      }
    }

    delete [] assignment2;
    delete [] received_edges;

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void print_tiles(std::string msg, int output_rank) {
    MPI_Barrier(MPI_COMM_WORLD);
    {
      if (global_myrank == output_rank) {
        std::cout << "Rank " << global_myrank << "\t" << msg << std::endl;
        for (int i = 0; i < ntiles_y; i++) {
          for (int j = 0; j < ntiles_x; j++) {
            std::cout << nodeIds[i + j * ntiles_y];
          }
          std::cout << std::endl;
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void Allocate2DPartitioned(int m, int n, int _num_tiles_x, int _num_tiles_y,
                             int (*_pfn)(int, int, int, int, int)) {
    num_tiles_x = _num_tiles_x;
    num_tiles_y = _num_tiles_y;
    pfn = _pfn;
    int vx, vy;
    int roundup = 256;
    int ntiles_x = num_tiles_x;
    int ntiles_y = num_tiles_y;
    vx = ((((n + ntiles_x - 1) / ntiles_x) + roundup - 1) / roundup) * roundup;
    vy = ((((m + ntiles_y - 1) / ntiles_y) + roundup - 1) / roundup) * roundup;

    int* nodeIds = reinterpret_cast<int*>(
        _mm_malloc(num_tiles_x * num_tiles_y * sizeof(int), 64));
    int* startx =
        reinterpret_cast<int*>(_mm_malloc((num_tiles_x + 1) * sizeof(int), 64));
    int* starty =
        reinterpret_cast<int*>(_mm_malloc((num_tiles_y + 1) * sizeof(int), 64));

    for (int i = 0; i < num_tiles_y; i++) {
      for (int j = 0; j < num_tiles_x; j++) {
        nodeIds[i + j * num_tiles_y] =
            pfn(j, i, num_tiles_x, num_tiles_y, global_nrank);
      }
    }

    for (int j = 0; j < num_tiles_x; j++) {
      startx[j] = std::min(vx * j, n);
    }

    for (int i = 0; i < num_tiles_y; i++) {
      starty[i] = std::min(vy * i, m);
    }
    startx[num_tiles_x] = n;
    starty[num_tiles_y] = m;

    set(m, n, num_tiles_x, num_tiles_y, nodeIds, startx, starty);
  }

  template<typename T>
  void setElement(const int idx, const int idy, T val)
  {
    assert(!empty);
    int ival, jval;
    int tile = getPartition(idy, idx, &ival, &jval);
    assert(tile != -1);
    if(nodeIds[ival + jval * ntiles_y] == global_myrank)
    {
      tiles[ival][jval].set(idx - start_idx[jval], idy - start_idy[ival], val);
    }
  }

  template<typename T>
  T getElement(const int idx, const int idy) const {
    assert(!empty);
    int ival, jval;
    int tile = getPartition(idy, idx, &ival, &jval); // Is this right?
    assert(tile != -1);
    if(nodeIds[ival + jval * ntiles_y] == global_myrank)
    {
      return tiles[ival][jval].get(idx - start_idx[jval], idy - start_idy[ival]);
    }
  }

  template <typename T>
  void get_edges(edgelist_t<T>* edgelist) const {
    // Get nnz
    int nnzs = 0;
    for (int i = 0; i < ntiles_y; i++) {
      for (int j = 0; j < ntiles_x; j++) {
        if (nodeIds[i + j * ntiles_y] == global_myrank) {
          nnzs += tiles[i][j].nnz;
        }
      }
    }
    edgelist->m = m;
    edgelist->n = n;
    edgelist->nnz = nnzs;
    if(nnzs > 0)
    {
      edgelist->edges = reinterpret_cast<edge_t<T>*>(
          _mm_malloc((uint64_t)nnzs * (uint64_t)sizeof(edge_t<T>), 64));

      nnzs = 0;
      for (int i = 0; i < ntiles_y; i++) {
        for (int j = 0; j < ntiles_x; j++) {
          if (nodeIds[i + j * ntiles_y] == global_myrank) {
            tiles[i][j]
                .get_edges(edgelist->edges + nnzs, start_idy[i], start_idx[j]);
            nnzs += tiles[i][j].nnz;
          }
        }
      }
    }
  }

  uint64_t getNNZ()
  {
    uint64_t total_nnz = 0;
    for(int i = 0 ; i < ntiles_y ; i++)
    {
      for(int j = 0 ; j < ntiles_x ; j++)
      {
        if(nodeIds[i + j * ntiles_y] == global_myrank)
        {
          total_nnz += tiles[i][j].nnz;
        }
      }
    }
    // global reduction
    MPI_Allreduce(MPI_IN_PLACE, &total_nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return total_nnz;
  }
};

#endif  // SRC_SPMAT_H_
