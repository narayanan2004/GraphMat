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
/* Narayanan Sundaram (Intel Corp.)
 *  * ******************************************************************************/

#ifndef EDGELIST_TRANSFORMATIONS_H_
#define EDGELIST_TRANSFORMATIONS_H_

#include "GMDP/utils/edgelist.h"

template <typename T>
void remove_selfedges(edgelist_t<T>* edgelist) {
  int new_nnz = 0;
  edgelist_t<T> new_edgelist(edgelist->m, edgelist->n, edgelist->nnz);
  for(int i = 0; i < edgelist->nnz; i++) {
    if (edgelist->edges[i].src != edgelist->edges[i].dst) {
      new_edgelist.edges[new_nnz] = edgelist->edges[i];
      new_nnz++;
    }
  }
  edgelist->clear();
  edgelist->edges = new_edgelist.edges;
  edgelist->nnz = new_nnz;
  edgelist->m = new_edgelist.m;
  edgelist->n = new_edgelist.n;
  return;
}

template<typename T>
bool compare_for_duplicates(const edge_t<T>& e1, const edge_t<T>& e2) {
  if (e1.src < e2.src) return true;
  else if (e1.src > e2.src) return false;
  if (e1.dst < e2.dst) return true;
  else return false;
}

template<typename T>
void sort_types(edgelist_t<T>* edgelist)
{
  __gnu_parallel::sort(edgelist->edges, edgelist->edges+edgelist->nnz, compare_for_duplicates<T>);
}

template <typename T>
void remove_duplicate_edges_local(edgelist_t<T>* edgelist) {
  if (edgelist->nnz > 0) {
    sort_types<T>(edgelist);
    edgelist_t<T> new_edgelist(edgelist->m, edgelist->n, edgelist->nnz);
    unsigned long int nnz2 = 0;
    new_edgelist.edges[0] = edgelist->edges[0]; 
    nnz2=1;

    for(unsigned long int i = 1; i < edgelist->nnz; i++) {
      if ((edgelist->edges[i].src == edgelist->edges[i-1].src) && 
          (edgelist->edges[i].dst == edgelist->edges[i-1].dst)) {
        continue;
      } else {
        new_edgelist.edges[nnz2] = edgelist->edges[i];
        nnz2++;
      }
    }
    edgelist->clear();
    edgelist->edges = new_edgelist.edges;
    edgelist->nnz = nnz2; 
    edgelist->m = new_edgelist.m;
    edgelist->n = new_edgelist.n;
  }
}

template <typename T>
void shuffle_edges(edgelist_t<T>* edgelist) {
  int m = edgelist->m;
  int n = edgelist->n;
  auto nnz_l = edgelist->nnz;
  int global_nrank = get_global_nrank();
  int global_myrank = get_global_myrank();


  unsigned long int new_nnz = 0;
  printf("Rank %d: Before shuffle %d edges\n", global_myrank, edgelist->nnz);
  edge_t<T> * tedges = new edge_t<T>[nnz_l];
  int* histogram = new int[omp_get_max_threads() * global_nrank]();
  int* offset = new int[omp_get_max_threads() * global_nrank]();
  int* woffset = new int[omp_get_max_threads() * global_nrank]();

  memset(histogram, 0, sizeof(int)*omp_get_max_threads() * global_nrank);
  memset(offset, 0, sizeof(int)*omp_get_max_threads() * global_nrank);
  memset(woffset, 0, sizeof(int)*omp_get_max_threads() * global_nrank);

  #pragma omp parallel 
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    auto points_per_thread = nnz_l/nthreads;
    auto start = tid*points_per_thread;
    auto end = start + points_per_thread;
    start = (start > nnz_l)?(nnz_l):(start);
    end = (end > nnz_l)?(nnz_l):(end);
    end = (tid == nthreads-1)?(nnz_l):(end);
    for(auto i = start ; i < end ; i++) {
      int bin = (edgelist->edges[i].src-1)%global_nrank;
      assert(bin >= 0 && bin <= global_nrank-1);
      histogram[tid*global_nrank + bin]+=1;
    }
  }
  offset[0] = 0;
  for (int bin = 0; bin < global_nrank; bin++) {
    for (int tid = 0; tid < omp_get_max_threads(); tid++) {
      if (tid > 0) {
        offset[tid*global_nrank + bin] = offset[(tid-1)*global_nrank + bin] + histogram[(tid-1)*global_nrank + bin];
      }
      if (tid == 0 && bin > 0) {
        offset[tid*global_nrank + bin] = offset[(omp_get_max_threads()-1)*global_nrank + bin-1] + histogram[(omp_get_max_threads()-1)*global_nrank + bin-1];
      }
    }
  }
  #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();
    auto points_per_thread = nnz_l/nthreads;
    auto start = tid*points_per_thread;
    auto end = start + points_per_thread;
    start = (start > nnz_l)?(nnz_l):(start);
    end = (end > nnz_l)?(nnz_l):(end);
    end = (tid == nthreads-1)?(nnz_l):(end);
    for(auto i = start ; i < end ; i++) {
      int bin = (edgelist->edges[i].src-1)%global_nrank;
      assert(bin >= 0 && bin <= global_nrank-1);
      tedges[offset[omp_get_thread_num()*global_nrank + bin] + woffset[omp_get_thread_num()*global_nrank + bin]] = edgelist->edges[i];
      woffset[omp_get_thread_num()*global_nrank + bin]++;
    }
  }

  unsigned long int * positions = new unsigned long[global_nrank+1];
  unsigned long int * counts = new unsigned long[global_nrank];
  unsigned long int * recv_positions = new unsigned long[global_nrank+1];
  unsigned long int * recv_counts = new unsigned long[global_nrank];
  for (int bin = 0; bin < global_nrank; bin++) {
    positions[bin] = offset[bin];
    counts[bin] = 0;
    for (int tid = 0; tid < omp_get_max_threads(); tid++) {
      counts[bin] += histogram[tid*global_nrank + bin];
    }
  }
  positions[global_nrank] = nnz_l;

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


  recv_positions[0] = 0;
  for(int i = 0 ; i < global_nrank ; i++)
  {
    new_nnz += recv_counts[i];
    recv_positions[i+1] = new_nnz;
  }

  MPI_Datatype MPI_EDGE_T;
  MPI_Type_contiguous(sizeof(edge_t<T>), MPI_CHAR, &MPI_EDGE_T);
  MPI_Type_commit(&MPI_EDGE_T);
  for (int i = 0; i < global_nrank; i++) {
    MPI_Isend(tedges + positions[i], counts[i] ,
        MPI_EDGE_T, i, global_myrank, MPI_COMM_WORLD, &mpi_req[i]);
  }
  auto received_edges = edgelist_t<T>(m, n, new_nnz);

  for (int i = 0; i < global_nrank; i++) {
    MPI_Irecv(received_edges.edges + recv_positions[i], recv_counts[i] ,
        MPI_EDGE_T, i, i, MPI_COMM_WORLD, &mpi_req[i+global_nrank]);
  }

  MPI_Waitall(2 * global_nrank, mpi_req, mpi_status);
  MPI_Barrier(MPI_COMM_WORLD);

  delete [] mpi_status;
  delete [] mpi_req;
  delete [] positions;
  delete [] counts;
  delete [] recv_positions;
  delete [] recv_counts;
  delete [] tedges;
  delete [] histogram;
  delete [] offset;
  delete [] woffset;

  printf("Rank %d: After shuffle %ld edges\n", global_myrank, new_nnz);

  edgelist->clear();
  edgelist->edges = received_edges.edges;
  edgelist->m = m;
  edgelist->n = n;
  edgelist->nnz = new_nnz;
  return;

}

template <typename T>
void remove_duplicate_edges(edgelist_t<T>* edgelist) {
// everyone shuffles data to others (disjoint sets based on src), then everyone performs updates locally.

    if(get_global_nrank() == 1)
    {
      remove_duplicate_edges_local(edgelist);
    }
    else
    {
      shuffle_edges(edgelist);
      remove_duplicate_edges_local(edgelist);
    }
    return;
}
/*      printf("Rank %d: Before shuffle %d edges\n", global_myrank, edgelist->nnz);
      edge_t<T> * tedges = new edge_t<T>[nnz_l];
      int* histogram = new int[omp_get_max_threads() * global_nrank]();
      int* offset = new int[omp_get_max_threads() * global_nrank]();
      int* woffset = new int[omp_get_max_threads() * global_nrank]();

      memset(histogram, 0, sizeof(int)*omp_get_max_threads() * global_nrank);
      memset(offset, 0, sizeof(int)*omp_get_max_threads() * global_nrank);
      memset(woffset, 0, sizeof(int)*omp_get_max_threads() * global_nrank);

      #pragma omp parallel 
      {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        auto points_per_thread = nnz_l/nthreads;
        auto start = tid*points_per_thread;
        auto end = start + points_per_thread;
        start = (start > nnz_l)?(nnz_l):(start);
        end = (end > nnz_l)?(nnz_l):(end);
        end = (tid == nthreads-1)?(nnz_l):(end);
        for(auto i = start ; i < end ; i++) {
          //int bin = (edgelist->edges[i].src-1)*global_nrank/n;
          int bin = (edgelist->edges[i].src-1)%global_nrank;
    assert(bin >= 0 && bin <= global_nrank-1);
          histogram[tid*global_nrank + bin]+=1;
        }
      }
      offset[0] = 0;
      for (int bin = 0; bin < global_nrank; bin++) {
        for (int tid = 0; tid < omp_get_max_threads(); tid++) {
    if (tid > 0) {
      offset[tid*global_nrank + bin] = offset[(tid-1)*global_nrank + bin] + histogram[(tid-1)*global_nrank + bin];
    }
    if (tid == 0 && bin > 0) {
      offset[tid*global_nrank + bin] = offset[(omp_get_max_threads()-1)*global_nrank + bin-1] + histogram[(omp_get_max_threads()-1)*global_nrank + bin-1];
    }
        }
      }
      #pragma omp parallel
      {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        auto points_per_thread = nnz_l/nthreads;
        auto start = tid*points_per_thread;
        auto end = start + points_per_thread;
        start = (start > nnz_l)?(nnz_l):(start);
        end = (end > nnz_l)?(nnz_l):(end);
        end = (tid == nthreads-1)?(nnz_l):(end);
        for(auto i = start ; i < end ; i++) {
          //int bin = (edgelist->edges[i].src-1)*global_nrank/n;
          int bin = (edgelist->edges[i].src-1)%global_nrank;
    assert(bin >= 0 && bin <= global_nrank-1);
    tedges[offset[omp_get_thread_num()*global_nrank + bin] + woffset[omp_get_thread_num()*global_nrank + bin]] = edgelist->edges[i];
          woffset[omp_get_thread_num()*global_nrank + bin]++;
        }
      }

      unsigned long int * positions = new unsigned long[global_nrank+1];
      unsigned long int * counts = new unsigned long[global_nrank];
      unsigned long int * recv_positions = new unsigned long[global_nrank+1];
      unsigned long int * recv_counts = new unsigned long[global_nrank];
      for (int bin = 0; bin < global_nrank; bin++) {
  positions[bin] = offset[bin];
        counts[bin] = 0;
        for (int tid = 0; tid < omp_get_max_threads(); tid++) {
    counts[bin] += histogram[tid*global_nrank + bin];
  }
      }
      positions[global_nrank] = nnz_l;
      
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
  
  
      recv_positions[0] = 0;
      for(int i = 0 ; i < global_nrank ; i++)
      {
        new_nnz += recv_counts[i];
        recv_positions[i+1] = new_nnz;
      }

      MPI_Datatype MPI_EDGE_T;
      MPI_Type_contiguous(sizeof(edge_t<T>), MPI_CHAR, &MPI_EDGE_T);
      MPI_Type_commit(&MPI_EDGE_T);
      for (int i = 0; i < global_nrank; i++) {
        MPI_Isend(tedges + positions[i], counts[i] ,
                  MPI_EDGE_T, i, global_myrank, MPI_COMM_WORLD, &mpi_req[i]);
      }
      auto received_edges = edgelist_t<T>(m, n, new_nnz);
  
      for (int i = 0; i < global_nrank; i++) {
        MPI_Irecv(received_edges.edges + recv_positions[i], recv_counts[i] ,
                  MPI_EDGE_T, i, i, MPI_COMM_WORLD, &mpi_req[i+global_nrank]);
      }
  
      MPI_Waitall(2 * global_nrank, mpi_req, mpi_status);
      MPI_Barrier(MPI_COMM_WORLD);

      delete [] mpi_status;
      delete [] mpi_req;
      delete [] positions;
      delete [] counts;
      delete [] recv_positions;
      delete [] recv_counts;
      delete [] tedges;
      delete [] histogram;
      delete [] offset;
      delete [] woffset;

      printf("Rank %d: After shuffle %ld edges\n", global_myrank, new_nnz);

      edgelist->clear();
      edgelist->edges = received_edges.edges;
      edgelist->m = m;
      edgelist->n = n;
      edgelist->nnz = new_nnz;
      remove_duplicate_edges_local(edgelist);
      return;
    }
}*/

template <typename T>
void randomize_edge_direction(edgelist_t<T>* edgelist) {
  for(int i = 0; i < edgelist->nnz; i++) {
    if ((double)rand()/(double)RAND_MAX < 0.5) {
      std::swap(edgelist->edges[i].src, edgelist->edges[i].dst);
    }
  }
}

template <typename T>
void create_bidirectional_edges(edgelist_t<T>* edgelist) {
  edgelist_t<T> new_edgelist(edgelist->m, edgelist->n, edgelist->nnz*2);
  for(int i = 0; i < edgelist->nnz; i++) {
    new_edgelist.edges[2*i] = edgelist->edges[i];
    new_edgelist.edges[2*i+1] = edgelist->edges[i];
    std::swap(new_edgelist.edges[2*i+1].src, new_edgelist.edges[2*i+1].dst);
  }
  edgelist->clear();
  edgelist->edges = new_edgelist.edges;
  edgelist->nnz = new_edgelist.nnz;
  edgelist->m = new_edgelist.m;
  edgelist->n = new_edgelist.n;
  return;
}

template <typename T>
void convert_to_dag(edgelist_t<T>* edgelist) {
  for(int i = 0; i < edgelist->nnz; i++) {
    if (edgelist->edges[i].src > edgelist->edges[i].dst) {
      std::swap(edgelist->edges[i].src, edgelist->edges[i].dst);
    }
  }
}

template <typename T>
void random_edge_weights(edgelist_t<T>* edgelist, int random_range) {
  for(int i = 0; i < edgelist->nnz; i++) {
    double t = ((double)rand()/(double)RAND_MAX*(double)random_range);  
    if (t > random_range) t = random_range;
    if (t < 1) t = 1;
    edgelist->edges[i].val = (T)t;
  }
}

template <typename T>
edgelist_t<T> filter_edges(edgelist_t<T>* edgelist, bool(*filter_function)(edge_t<T>, void*), void* param=NULL) {
  edgelist_t<T> new_edgelist(edgelist->m, edgelist->n, edgelist->nnz);
  int k = 0;
  for(int i = 0; i < edgelist->nnz; i++) {
    if (filter_function(edgelist->edges[i], param)) {
      new_edgelist.edges[k] = edgelist->edges[i];
      k++;
    }
  }
  new_edgelist.nnz = k;
  return new_edgelist;
}


#endif
