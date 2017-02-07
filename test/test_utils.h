/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ******************************************************************************/
/* Narayanan Sundaram (Intel Corp.), Michael Anderson (Intel Corp.)
 * ******************************************************************************/
#ifndef _TEST_UTILS_H_
#define _TEST_UTILS_H_

#include <iostream>
#include "catch.hpp"
#include <algorithm>

template<typename T>
bool edge_compare(const GraphMat::edge_t<T> &e1,
                  const GraphMat::edge_t<T> &e2)
{
        if( (e1.src < e2.src) ||
            ((e1.src == e2.src) && (e1.dst < e2.dst)) ||
            ((e1.src == e2.src) && (e1.dst == e2.dst) && (e1.val < e2.val)) )
        {
                return true;
        }
        return false;
}

template <typename EDGE_T>
void collect_edges(const GraphMat::edgelist_t<EDGE_T>& in_edges, GraphMat::edgelist_t<EDGE_T>& out_edges) {

    REQUIRE(sizeof(EDGE_T)%sizeof(int) == 0);
    int T_by_int = sizeof(in_edges.edges[0])/sizeof(int);

    int* OERecvCount = new int[GraphMat::get_global_nrank()];
    MPI_Allgather(&in_edges.nnz, 1, MPI_INT, OERecvCount, 1, MPI_INT, MPI_COMM_WORLD);

    int* OERecvOffset = new int[GraphMat::get_global_nrank()];
    int* OERecvCountInt = new int[GraphMat::get_global_nrank()];
    OERecvOffset[0] = 0;
    for (int i = 1; i < GraphMat::get_global_nrank(); i++) {
      OERecvOffset[i] = OERecvOffset[i-1] + T_by_int*OERecvCount[i-1];      
    }
    for (int i = 0; i < GraphMat::get_global_nrank(); i++) {
      OERecvCountInt[i] = T_by_int*OERecvCount[i];
    }

    int nnz = 0;
    for (int i = 0; i < GraphMat::get_global_nrank(); i++) {
      nnz += OERecvCount[i];
    }
    out_edges = GraphMat::edgelist_t<EDGE_T>(in_edges.m, in_edges.n, nnz);

    MPI_Allgatherv(in_edges.edges, in_edges.nnz*T_by_int, MPI_INT, out_edges.edges, OERecvCountInt, OERecvOffset, MPI_INT, MPI_COMM_WORLD);

    delete [] OERecvCount;
    delete [] OERecvCountInt;
    delete [] OERecvOffset;
}

template <typename EDGE_T>
void distribute_edges(const int root_rank, const GraphMat::edgelist_t<EDGE_T>& in_edges, GraphMat::edgelist_t<EDGE_T>& out_edges) {

      int global_nrank = GraphMat::get_global_nrank();
      int global_myrank = GraphMat::get_global_myrank();

      unsigned long int * positions = new unsigned long[global_nrank+1];
      unsigned long int * counts = new unsigned long[global_nrank];
      unsigned long int * recv_positions = new unsigned long[global_nrank+1];
      unsigned long int * recv_counts = new unsigned long[global_nrank];

      memset(positions, 0, sizeof(unsigned long)*(global_nrank+1));
      memset(counts, 0, sizeof(unsigned long)*(global_nrank));

      if (global_myrank == root_rank) {
        int points_per_rank = in_edges.nnz/global_nrank;
        int remaining = in_edges.nnz - (global_nrank * points_per_rank);
        assert(remaining < global_nrank);
        for (int bin = 0; bin < global_nrank; bin++) {
          counts[bin] = points_per_rank;
          if (remaining > 0) {
            counts[bin]++;
            remaining--;
          }
        }

        for (int bin = 1; bin < global_nrank; bin++) {
	  positions[bin] = positions[bin-1] + counts[bin-1];
        }
        positions[global_nrank] = in_edges.nnz;
      }
      
      MPI_Barrier(MPI_COMM_WORLD);
  
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
  
      auto new_nnz = 0; 
      recv_positions[0] = 0;
      for(int i = 0 ; i < global_nrank ; i++)
      {
        new_nnz += recv_counts[i];
        recv_positions[i+1] = new_nnz;
      }

      MPI_Datatype MPI_EDGE_T;
      MPI_Type_contiguous(sizeof(GraphMat::edge_t<EDGE_T>), MPI_CHAR, &MPI_EDGE_T);
      MPI_Type_commit(&MPI_EDGE_T);
      for (int i = 0; i < global_nrank; i++) {
        MPI_Isend(in_edges.edges + positions[i], counts[i] ,
                  MPI_EDGE_T, i, global_myrank, MPI_COMM_WORLD, &mpi_req[i]);
      }
      out_edges = GraphMat::edgelist_t<EDGE_T>(in_edges.m, in_edges.n, new_nnz);
  
      for (int i = 0; i < global_nrank; i++) {
        MPI_Irecv(out_edges.edges + recv_positions[i], recv_counts[i] ,
                  MPI_EDGE_T, i, i, MPI_COMM_WORLD, &mpi_req[i+global_nrank]);
      }
  
      MPI_Waitall(2 * global_nrank, mpi_req, mpi_status);
      std::cout << "Rank " << global_myrank << " has " << new_nnz << " edges after distribute" << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);

      delete [] mpi_status;
      delete [] mpi_req;
      delete [] positions;
      delete [] counts;
      delete [] recv_positions;
      delete [] recv_counts;

}

template <typename T>
void mul(T a, T b, T * c, void* vsp) {*c = a*b;}

template <typename T>
void add(T a, T b, T * c, void* vsp) {*c = a+b;}

template <typename T>
void max(T a, T b, T * c, void* vsp) {*c = std::max(a,b);}

template <typename T>
void min(T a, T b, T * c, void* vsp) {*c = std::min(a,b);}

#endif
