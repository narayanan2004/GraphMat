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


#ifndef SRC_SPVEC_H_
#define SRC_SPVEC_H_

#include <string>
#include <algorithm>
#include <vector>

#include "GMDP/vectors/DenseSegment.h"

template <typename SpSegment>
class SpVec {
 public:
  std::string name;
  int nsegments;
  int n;
  int num_tiles_x;
  int global_nrank, global_myrank;

  std::vector<int> nodeIds;
  std::vector<int> start_id;
  std::vector<SpSegment*> segments;

  friend boost::serialization::access;
  template<class Archive> 
  void serialize(Archive& ar, const unsigned int version) {
    ar & name;
    ar & nsegments;
    ar & n;
    ar & num_tiles_x;
    ar & global_nrank;
    ar & global_myrank;
    ar & nodeIds;
    ar & start_id;
    ar & segments;
  }

  SpVec() {};
  SpVec(int _n, int _num_tiles_x,
                           int (*_pfn)(int, int, int)) {

    global_nrank = get_global_nrank();
    global_myrank = get_global_myrank(); 
    num_tiles_x = _num_tiles_x;
    n = _n;
    int vx, vy;
    int roundup = 256;
    nsegments = num_tiles_x;
    vx =
        ((((n + nsegments - 1) / nsegments) + roundup - 1) / roundup) * roundup;

    // In case the roundup affected the num tiles
    for (int j = 0; j < num_tiles_x; j++) {
      nodeIds.push_back(_pfn(j, num_tiles_x, global_nrank));
    }
    for (int j = 0; j < num_tiles_x; j++) {
      start_id.push_back(std::min(vx * j, n));
    }
    start_id.push_back(n);

    // Copy metadata
    assert(nsegments > 0);

    // Allocate space for tiles
    for (int j = 0; j < nsegments; j++) {
      segments.push_back(new SpSegment(start_id[j + 1] - start_id[j]));
    }
  }

  ~SpVec() 
  {
    for(auto it = segments.begin() ; it != segments.end() ; it++)
    {
      delete *it;
    }
    segments.clear();
  }

  inline int getPartition(int src) const {
    for (int i = 0; i < nsegments; i++) {
      if ((src > start_id[i]) && (src <= start_id[i + 1])) {
        return i;
      }
    }
    return -1;
  }

  template <typename T>
  void get_edges(edgelist_t<T> * blob) const
  {
    blob->nnz = 0;
    blob->m = n;
    blob->n = 1;
    for(int segment = 0 ; segment < nsegments ; segment++)
    {
      if(nodeIds[segment] == global_myrank)
      {
        blob->nnz += segments[segment]->compute_nnz();
      }
    }
    if(blob->nnz > 0)
    {
      blob->edges = reinterpret_cast<edge_t<T>*>(
        _mm_malloc((uint64_t)blob->nnz * (uint64_t)sizeof(edge_t<T>), 64));
      unsigned int nnzs = 0;
      for(int segment = 0 ; segment < nsegments ; segment++)
      {
        if(nodeIds[segment] == global_myrank)
        {
          segments[segment]->get_edges(blob->edges + nnzs, start_id[segment]);
          nnzs += segments[segment]->compute_nnz();
        }
      }
    }
  }

  // Note: replace with all-to-all-v
  template <typename T>
  void ingestEdgelist(edgelist_t<T> blob) {
    int nnz_l = blob.nnz;
    edge_t<T>* edge_list = blob.edges;

    int m = blob.m;
    assert(blob.n == 1);

    printf("Rank %d: Before shuffle %d edges\n", global_myrank, blob.nnz);

    // Done with partitioning
    // Now, assign.
    int* assignment = new int[nnz_l];
#pragma omp parallel for
    for (int i = 0; i < nnz_l; i++) {
      int tile = getPartition(edge_list[i].src);
      assert(tile != -1);
      assignment[i] = nodeIds[tile];
    }
    // assignment over
    MPI_Barrier(MPI_COMM_WORLD);

    // pack into messages
    // calculate message sizes
    int* count = new int[global_nrank];
    int* recv_count = new int[global_nrank];
    MPI_Request* mpi_req = new MPI_Request[2 * global_nrank];
    MPI_Status* mpi_status = new MPI_Status[2 * global_nrank];
    memset(count, 0, sizeof(int) * global_nrank);

    for (int i = 0; i < nnz_l; i++) {
      int r = assignment[i];
      count[r]++;
    }
    for (int i = 0; i < global_nrank; i++) {
      MPI_Isend(&count[i], 1, MPI_INT, i, global_myrank, MPI_COMM_WORLD,
                &mpi_req[i]);
    }
    for (int i = 0; i < global_nrank; i++) {
      MPI_Irecv(&recv_count[i], 1, MPI_INT, i, i, MPI_COMM_WORLD,
                &mpi_req[i + global_nrank]);
    }
    MPI_Waitall(2 * global_nrank, mpi_req, mpi_status);
    MPI_Barrier(MPI_COMM_WORLD);

    // pack the messages and send
    edge_t<T>** msg = new edge_t<T>* [global_nrank];
    int* offsets = new int[global_nrank];
    for (int i = 0; i < global_nrank; i++) {
      msg[i] = new edge_t<T>[count[i]];
      offsets[i] = 0;
    }
    for (int i = 0; i < nnz_l; i++) {
      int r = assignment[i];
      msg[r][offsets[r]] = edge_list[i];
      ++offsets[r];
    }
    for (int i = 0; i < global_nrank; i++) {
      MPI_Isend(msg[i], (uint64_t)sizeof(edge_t<T>) * (uint64_t)count[i],
                MPI_CHAR, i, global_myrank, MPI_COMM_WORLD, &mpi_req[i]);
    }

    // receive messages into final_edge_list
    int new_nnz = 0;
    int* local_hist = new int[global_nrank + 1];
    local_hist[0] = 0;
    for (int i = 0; i < global_nrank; i++) {
      new_nnz += recv_count[i];
      local_hist[i + 1] = local_hist[i] + recv_count[i];
    }
    edge_t<T>* final_edge_list = reinterpret_cast<edge_t<T>*>(
        _mm_malloc((uint64_t)new_nnz * (uint64_t)sizeof(edge_t<T>), 64));
    for (int i = 0; i < global_nrank; i++) {
      MPI_Irecv(&final_edge_list[local_hist[i]],
                (uint64_t)sizeof(edge_t<T>) * (uint64_t)recv_count[i], MPI_CHAR,
                i, i, MPI_COMM_WORLD, &mpi_req[i + global_nrank]);
    }
    MPI_Waitall(2 * global_nrank, mpi_req, mpi_status);

    for (int i = 0; i < global_nrank; i++) {
      delete[] msg[i];
    }
    delete[] msg;
    delete[] local_hist;
    delete[] offsets;
    delete[] count;
    delete[] recv_count;
    delete[] mpi_req;
    delete[] mpi_status;

    printf("Rank %d: After shuffle %d edges\n", global_myrank, new_nnz);

    for (int i = 0; i < new_nnz; i++) {
      int ival, jval;
      int tile = getPartition(final_edge_list[i].src);
      assert(tile != -1);
      assert(nodeIds[tile] == global_myrank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Sort these edges by segment ID
    edge_t<T>* edges = reinterpret_cast<edge_t<T>*>(
        _mm_malloc((uint64_t)new_nnz * (uint64_t)sizeof(edge_t<T>), 64));
    int* partitions = reinterpret_cast<int*>(
        _mm_malloc((uint64_t)new_nnz * (uint64_t)sizeof(int), 64));
    uint64_t* counts = reinterpret_cast<uint64_t*>(
        _mm_malloc((nsegments) * sizeof(uint64_t), 64));
    uint64_t* start_nzs = reinterpret_cast<uint64_t*>(
        _mm_malloc((nsegments + 1) * sizeof(uint64_t), 64));
    memset(counts, 0, (nsegments) * sizeof(uint64_t));
    memset(start_nzs, 0, (nsegments+1) * sizeof(uint64_t));
    for (uint64_t i = 0; i < (uint64_t)new_nnz; i++) {
      partitions[i] = getPartition(final_edge_list[i].src);
      counts[partitions[i]]++;
    }
    uint64_t acc = 0;
    for (int i = 0; i < nsegments; i++) {
      start_nzs[i] = acc;
      acc += counts[i];
    }
    start_nzs[nsegments] = acc;
    memset(counts, 0, (nsegments) * sizeof(uint64_t));
    for (uint64_t i = 0; i < (uint64_t)new_nnz; i++) {
      int new_idx = start_nzs[partitions[i]] + counts[partitions[i]];
      assert(new_idx < new_nnz);
      assert(new_idx >= 0);
      assert(partitions[i] < nsegments);
      assert(partitions[i] >= 0);
      edges[new_idx] = final_edge_list[i];
      counts[partitions[i]]++;
    }
    if(new_nnz > 0)
    {
      _mm_free(final_edge_list);
      _mm_free(partitions);
    }


    for (int segment_i = 0; segment_i < nsegments; segment_i++) {
      if (nodeIds[segment_i] == global_myrank) {
        int tile_m = start_id[segment_i + 1] - start_id[segment_i];
        int nnz = counts[segment_i];
        int start_nz = start_nzs[segment_i];
	assert(start_nz <= new_nnz);
	assert(nnz <= new_nnz);
        if(nnz > 0)
        {
              segments[segment_i]->ingestEdges(edges + start_nz, tile_m, nnz, start_id[segment_i]);
        }
      }
    }

    _mm_free(counts);
    _mm_free(start_nzs);
    _mm_free(edges);

    MPI_Barrier(MPI_COMM_WORLD);
  }


  template<typename T>
  void set(int idx, T val) {
    int partitionId = getPartition(idx);
    assert(partitionId >= 0);
    if (nodeIds[partitionId] == global_myrank) {
      assert(segments[partitionId]->capacity > 0);
      segments[partitionId]->set(idx - start_id[partitionId], val);
    }
  }

  void unset(int idx) {
    int partitionId = getPartition(idx);
    assert(partitionId >= 0);
    if (nodeIds[partitionId] == global_myrank) {
      assert(segments[partitionId]->capacity > 0);
      segments[partitionId]->unset(idx - start_id[partitionId]);
    }
  }

  template<typename T>
  void setAll(T val) {
    for(int segmentId = 0 ; segmentId < nsegments ; segmentId++)
    {
      if(nodeIds[segmentId] == global_myrank)
      {
        segments[segmentId]->setAll(val);
      }
    }
  }

  template<typename T>
  void get(const int idx, T * myres) const {
    int partitionId = getPartition(idx);
    assert(partitionId >= 0);
    if (nodeIds[partitionId] == global_myrank) {
      SpSegment * segment = segments[partitionId];
      *myres = segment->get(idx - start_id[partitionId]);
    }
  }

  int getNNZ()
  {
    int total_nnz = 0;
    for(int s = 0 ; s < nsegments ; s++)
    {
      if(nodeIds[s] == global_myrank)
      {
        //total_nnz += segments[s].getNNZ();
        total_nnz += segments[s]->compute_nnz();
      }
    }
    // global reduction
    MPI_Allreduce(MPI_IN_PLACE, &total_nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return total_nnz;
  }

  bool node_owner(const int idx) const {
    int partitionId = getPartition(idx);
    assert(partitionId >= 0);
    bool v;
    if (nodeIds[partitionId] == global_myrank) {
      v = true;
    } else {
      v = false;
    }
    return v;
  }

  void save(std::string fname, bool includeHeader ) const {
    for(int segment = 0 ; segment < nsegments ; segment++)
    {
      if(nodeIds[segment] == global_myrank)
      {
        segments[segment]->save(fname + std::to_string(segment), start_id[segment], n, includeHeader);
      }
    }
  }

};

#endif  // SRC_SPVEC_H_
