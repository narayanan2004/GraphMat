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


#ifndef SRC_DENSESEGMENT_H_
#define SRC_DENSESEGMENT_H_

#include <string>
#include "src/edgelist.h"
#include "src/bitvector.h"


template <typename T>
class segment_props
{
  public:
    T * value;
    int * bit_vector;
    bool allocated;
    bool uninitialized;
    int nnz;

    T * compressed_data;
    int * compressed_indices;

    segment_props()
    {
      value = NULL;
      bit_vector=NULL;
      allocated=false;
      uninitialized=true;
      nnz=0;
    }

    void alloc(int capacity, int num_ints)
    {
      if(!allocated)
      {
        value = reinterpret_cast<T*>(_mm_malloc(capacity * sizeof(T) + num_ints*sizeof(int), 64));
        bit_vector = reinterpret_cast<int*>( value + capacity);
        compressed_data = reinterpret_cast<T*>(_mm_malloc(capacity * sizeof(T) + capacity*sizeof(int), 64));
        allocated = true;
        uninitialized = true;
      }
    }
};

template <typename T>
class DenseSegment {
 public:
  std::string name;
  int capacity;
  int num_ints;

  segment_props<T> properties;
  std::vector<segment_props<T> > to_be_received_properties;
  std::vector<segment_props<T> > received_properties;
  std::vector<segment_props<T> > uninitialized_properties;

  DenseSegment()
  {
    capacity=0;
    num_ints=0;
  }

  DenseSegment(int n) {
    capacity = n;
    num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
    //std::cout << "Capacity, num_ints: " << capacity << "\t" << num_ints << std::endl;
  }

  DenseSegment(edge_t<T>* edges, int _m, int _nnz, int row_start)
  {
    capacity = _m;
    num_ints = (capacity + sizeof(int) * 8 - 1) / (sizeof(int) * 8);

    alloc();
    initialize();

    for (uint64_t i = 0; i < (uint64_t)_nnz; i++) {
      int src = edges[i].src - row_start - 1;
      set_bitvector(src, properties.bit_vector);
      properties.value[src] = edges[i].val;
    }
    properties.nnz = _nnz;
    properties.uninitialized = false;
  }

  ~DenseSegment() {}

  int compute_nnz()
  {
    if(properties.uninitialized) return 0;
    int len = 0;
    #pragma omp parallel for reduction(+:len)
    for (int ii = 0 ; ii < num_ints ; ii++) {
      int p = _popcnt32(properties.bit_vector[ii]);
      len += p;
    }
    return len;
  }
  
  int compute_nnz(int start, int finish)
  {
    if(properties.uninitialized) return 0;
    int len = 0;
    #pragma omp parallel for reduction(+:len)
    for (int ii = start  ; ii < finish ; ii++) {
      int p = _popcnt32(properties.bit_vector[ii]);
      len += p;
    }
    return len;
  }

  bool should_compress(int test_nnz)
  {
    if(test_nnz > COMPRESSION_THRESHOLD * capacity)
    {
      return false;
    }
    return true;
  }

  void compress()
  {
    if(!should_compress(properties.nnz))
    {
      return;
    }
    assert(properties.allocated);
    assert(!properties.uninitialized);
    properties.compressed_indices = reinterpret_cast<int*>(properties.compressed_data + properties.nnz);

    int npartitions = omp_get_max_threads() * 16;
    int * partition_nnz = new int[npartitions];
    int * partition_nnz_scan = new int[npartitions+1];
    #pragma omp parallel for
    for(int p = 0 ; p < npartitions ; p++)
    {
      int i_per_partition = (num_ints + npartitions - 1) / npartitions;
      int start_i = i_per_partition * p;
      int end_i = i_per_partition * (p+1);
      if(end_i > num_ints) end_i = num_ints;
      partition_nnz[p] = compute_nnz(start_i, end_i);
    }
    partition_nnz_scan[0] = 0;
    properties.nnz = 0;
    for(int p = 0 ; p < npartitions ; p++)
    {
      partition_nnz_scan[p+1] = partition_nnz_scan[p] + partition_nnz[p];
      properties.nnz += partition_nnz[p];
    }

    #pragma omp parallel for
    for(int p = 0 ; p < npartitions ; p++)
    {
      int i_per_partition = (num_ints + npartitions - 1) / npartitions;
      int start_i = i_per_partition * p;
      int end_i = i_per_partition * (p+1);
      if(end_i > num_ints) end_i = num_ints;
      int nzcnt = partition_nnz_scan[p];
      
      for(int ii = start_i ; ii < end_i ; ii++)
      {
        if(_popcnt32(properties.bit_vector[ii]) == 0) continue;
        for(int i = ii*32 ; i < (ii+1)*32 ; i++)
        {
          if(get_bitvector(i, properties.bit_vector))
          {
            properties.compressed_data[nzcnt] = properties.value[i];
            properties.compressed_indices[nzcnt] = i;
            nzcnt++;
          }
        }
      }
    }
    delete [] partition_nnz;
    delete [] partition_nnz_scan;
  }

  void decompress()
  {
    if(!should_compress(properties.nnz))
    {
      return;
    }
    assert(properties.allocated);
    memset(properties.bit_vector, 0, num_ints* sizeof(int));
    properties.compressed_indices = reinterpret_cast<int*>(properties.compressed_data + properties.nnz);
    int npartitions = omp_get_max_threads();

    int * start_nnzs = new int[npartitions];
    int * end_nnzs = new int[npartitions];

    int mystart = 0;
    int my_nz_per = (properties.nnz + npartitions - 1) / npartitions;
    my_nz_per = ((my_nz_per + 31) / 32) * 32;
    for(int p = 0 ; p < npartitions ; p++)
    {
      start_nnzs[p] = mystart;
      mystart += my_nz_per;
      if(mystart > properties.nnz) mystart = properties.nnz;
      if(mystart < properties.nnz)
      {
        int start32 = properties.compressed_indices[mystart] / 32;
        while((mystart < properties.nnz) && properties.compressed_indices[mystart] / 32  == start32) mystart++;
      }
      end_nnzs[p] = mystart;
    }

    #pragma omp parallel for
    for(int p = 0 ; p < npartitions ; p++)
    {
      int start_nnz = start_nnzs[p];
      int end_nnz = end_nnzs[p];
      for(int i = start_nnz  ; i < end_nnz ; i++)
      {
        int idx = properties.compressed_indices[i];
        set_bitvector(idx, properties.bit_vector);
        properties.value[idx] = properties.compressed_data[i];
      }
    }

    delete [] start_nnzs;
    delete [] end_nnzs;
  }


  void set_uninitialized_received()
  {
    for(auto it = received_properties.begin() ; it != received_properties.end() ; it++)
    {
      it->uninitialized = true;
      uninitialized_properties.push_back(*it);
    }
    received_properties.clear();
  }

  void set_uninitialized() {
    set_uninitialized_received();
    properties.uninitialized = true;
    properties.nnz = 0;
  }

  void dealloc_received()
  {
    for(auto it = received_properties.begin() ; it != received_properties.end() ; it++)
    {
      if(it->allocated)
      {
        _mm_free(it->value);
      }
    }
    received_properties.clear();
  }

  void dealloc() {
    dealloc_received();
    if (properties.allocated) {
      _mm_free(properties.value);
      properties.allocated = false;
      properties.uninitialized = true;
      properties.nnz = 0;
    }
  }

  void alloc() {
    if(!properties.allocated)
    {
      properties.alloc(capacity, num_ints);
    }
  }

  void initialize()
  {
    if(properties.uninitialized)
    {
      memset(properties.bit_vector, 0, num_ints* sizeof(int));
    }
    properties.uninitialized = false;
  }

  int getNNZ()
  {
    return properties.nnz;
  }

  void set(int idx, T val) {
    alloc();
    initialize();
    if(!get_bitvector(idx-1, properties.bit_vector)) properties.nnz++;
    properties.value[idx - 1] = val;
    set_bitvector(idx-1, properties.bit_vector);
    properties.uninitialized = false;
  }

  void setAll(T val) {
    alloc();
    //initialize();
    properties.uninitialized=false;
    if(num_ints == 0) return;
    properties.bit_vector[num_ints-1] = 0;
    #pragma omp parallel for
    for(int i = 0 ; i < num_ints-1 ; i++)
    {
      properties.bit_vector[i] = 0xFFFFFFFF;
    }

    for(int idx = std::max(0, capacity-32) ; idx < capacity ; idx++)
    {
      set_bitvector(idx, properties.bit_vector);
    }
    properties.nnz = capacity;

    #pragma omp parallel for
    for(int i = 0 ; i < capacity ; i++)
    {
      properties.value[i] = val;
    }
  }

  T get(const int idx) const {
    assert(properties.allocated);
    assert(!properties.uninitialized);
    return properties.value[idx - 1];
  }

  void send_tile_metadata(int myrank, int dst_rank, int output_rank, std::vector<MPI_Request>* requests) {
    MPI_Send(&(properties.nnz), 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
  }

  void recv_tile_metadata(int myrank, int src_rank, int output_rank,
                 std::vector<MPI_Request>* requests) {

    // if uninitialized
    if(properties.uninitialized)
    {
      MPI_Recv(&(properties.nnz), 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
    else
    {
      // Create new buffer
      segment_props<T> new_properties;
      if(uninitialized_properties.size() > 0) 
      {
        new_properties = uninitialized_properties.back();
        uninitialized_properties.pop_back();
      }
      MPI_Recv(&(new_properties.nnz), 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      to_be_received_properties.push_back(new_properties);
    }
  }

  void send_tile(int myrank, int dst_rank, int output_rank, std::vector<MPI_Request>* requests) {
    MPI_Request r1;
    MPI_Isend(properties.value, capacity * sizeof(T) + num_ints * sizeof(int), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD,
               &r1);
    requests->push_back(r1);
  }

  void recv_tile(int myrank, int src_rank, int output_rank,
                 std::vector<MPI_Request>* requests) {

    alloc();
    if(properties.uninitialized)
    {
      MPI_Request r1;
      MPI_Irecv(properties.value, capacity * sizeof(T) + num_ints * sizeof(int), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r1);
      requests->push_back(r1);
      properties.uninitialized = false;
    }
    else
    {
      segment_props<T> new_properties = to_be_received_properties[0];
      to_be_received_properties.erase(to_be_received_properties.begin());
      new_properties.alloc(capacity, num_ints);
      MPI_Request r1;
      MPI_Irecv(new_properties.value, capacity * sizeof(T) + num_ints* sizeof(int), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r1);
      requests->push_back(r1);
      new_properties.uninitialized = false;
      received_properties.push_back(new_properties);
    }
  }


  void send_tile_compressed(int myrank, int dst_rank, int output_rank, std::vector<MPI_Request>* requests) {
    if(!should_compress(properties.nnz))
    {
      send_tile(myrank, dst_rank, output_rank, requests);
      return;
    }
    MPI_Request r1;
    MPI_Isend(properties.compressed_data, properties.nnz * sizeof(T) + properties.nnz * sizeof(int), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD,
               &r1);
    requests->push_back(r1);
  }

  void recv_tile_compressed(int myrank, int src_rank, int output_rank,
                 std::vector<MPI_Request>* requests) {

    alloc();
    if(properties.uninitialized)
    {
      if(!should_compress(properties.nnz))
      {
        recv_tile(myrank, src_rank, output_rank, requests);
        return;
      }
      MPI_Request r1;
      MPI_Irecv(properties.compressed_data, properties.nnz * sizeof(T) + properties.nnz * sizeof(int), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r1);
      requests->push_back(r1);
      properties.uninitialized = false;
    }
    else
    {
      segment_props<T> new_properties = to_be_received_properties[0];
      if(!should_compress(new_properties.nnz))
      {
        recv_tile(myrank, src_rank, output_rank, requests);
        return;
      }
      to_be_received_properties.erase(to_be_received_properties.begin());
      new_properties.alloc(capacity, num_ints);
      MPI_Request r1;
      MPI_Irecv(new_properties.compressed_data, new_properties.nnz * sizeof(T) + new_properties.nnz * sizeof(int), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r1);
      requests->push_back(r1);
      new_properties.uninitialized = false;
      received_properties.push_back(new_properties);
    }
  }

  void save(std::string fname, int start_id, int _m, bool includeHeader)
  {
    int nnz = compute_nnz();
    std::ofstream fout;
    fout.open(fname);
    if(includeHeader)
    {
      fout << _m << " " << nnz << std::endl;
    }
    for(int i = 0 ; i < capacity ; i++)
    {
      if(get_bitvector(i, properties.bit_vector))
      {
        fout << i + start_id << " " << properties.value[i] << std::endl;
      }
    }
    fout.close();
  }

  void get_edges(edge_t<T> * edges, unsigned int start_nz)
  {
    unsigned int mycnt = 0;
    for(int i = 0 ; i < capacity ; i++)
    {
      if(get_bitvector(i, properties.bit_vector))
      {
        edges[mycnt].src = start_nz + mycnt + 1;
        edges[mycnt].val = properties.value[i];
        mycnt++;
      }
    }
  }
};
#endif  // SRC_DENSESEGMENT_H_
