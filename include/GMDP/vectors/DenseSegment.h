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

#include "GMDP/utils/edgelist.h"
#include "GMDP/utils/bitvector.h"
#include "GMDP/singlenode/unionreduce.h"

#include <string>
#include <vector>
#include <sstream>
#include <cstdio>

inline double get_compression_threshold();

enum compression_decision
{
  NONE,
  COMPRESSED,
  SERIALIZED
};

struct send_metadata
{
  int nnz;
  size_t serialized_nbytes;
  size_t serialized_npartitions;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
      ar & nnz;
      ar & serialized_nbytes;
      ar & serialized_npartitions;
  }
};

template <typename T>
class buffer
{
  public:
    bool uninitialized;
    int nnz;
    int capacity;
    int num_ints;
    size_t serialized_nbytes;
    size_t serialized_npartitions;

    T * value;
    int * bit_vector;
    T * compressed_data;
    int * compressed_indices;
    char * serialized_data;
    size_t * serialized_partition_nbytes_scan;
    size_t * serialized_partition_nnz_scan;

    // Serialize
    friend boost::serialization::access;
    template<class Archive> 
    void save(Archive& ar, const unsigned int version) const {
      ar & uninitialized;
      ar & nnz;
      ar & capacity;
      ar & num_ints;
      ar & serialized_nbytes;
      ar & serialized_npartitions;
      for(int i = 0 ; i < capacity; i++)
      {
        ar & value[i];
      }
      for(int i = 0 ; i < num_ints; i++)
      {
        ar & bit_vector[i];
      }
      for(int i = 0 ; i < capacity ; i++)
      {
        ar & compressed_data[i];
      }
      for(int i = 0 ; i < capacity ; i++)
      {
        ar & compressed_indices[i];
      }
      for(int i = 0 ; i < serialized_nbytes; i++)
      {
        ar & serialized_data[i];
      }
      for(int i = 0 ; i < serialized_npartitions + 1; i++)
      {
        ar & serialized_partition_nbytes_scan[i];
      }
      for(int i = 0 ; i < serialized_npartitions + 1; i++)
      {
        ar & serialized_partition_nnz_scan[i];
      }
    }
    template<class Archive> 
    void load(Archive& ar, const unsigned int version) {
      ar & uninitialized;
      ar & nnz;
      ar & capacity;
      ar & num_ints;
      ar & serialized_nbytes;
      ar & serialized_npartitions;
      delete [] value;
      delete [] bit_vector;
      delete [] compressed_data;
      delete [] compressed_indices;
      delete [] serialized_data;
      delete [] serialized_partition_nbytes_scan;
      delete [] serialized_partition_nnz_scan;
      value = new T[capacity];
      bit_vector = new int[num_ints];
      compressed_data = new T[capacity];
      compressed_indices = new int[capacity];
      serialized_data = new char[serialized_nbytes];
      serialized_partition_nbytes_scan = new size_t[serialized_npartitions+1];
      serialized_partition_nnz_scan = new size_t[serialized_npartitions+1];
      for(int i = 0 ; i < capacity; i++)
      {
        ar & value[i];
      }
      for(int i = 0 ; i < num_ints; i++)
      {
        ar & bit_vector[i];
      }
      for(int i = 0 ; i < capacity ; i++)
      {
        ar & compressed_data[i];
      }
      for(int i = 0 ; i < capacity ; i++)
      {
        ar & compressed_indices[i];
      }
      for(int i = 0 ; i < serialized_nbytes; i++)
      {
        ar & serialized_data[i];
      }
      for(int i = 0 ; i < serialized_npartitions + 1; i++)
      {
        ar & serialized_partition_nbytes_scan[i];
      }
      for(int i = 0 ; i < serialized_npartitions + 1; i++)
      {
        ar & serialized_partition_nnz_scan[i];
      }
    }
    BOOST_SERIALIZATION_SPLIT_MEMBER()

    buffer(int _capacity, int _num_ints)
    {
      capacity = _capacity;
      num_ints = _num_ints;
      value = new T[capacity]; 
      bit_vector = new int[num_ints];
      //compressed_data = reinterpret_cast<T*>(_mm_malloc(capacity * sizeof(T) + capacity*sizeof(int), 64));
      compressed_data = new T[capacity];
      compressed_indices = new int[capacity];
      uninitialized = true;

      serialized_data = new char[0];
      serialized_nbytes = 0;
      serialized_npartitions = omp_get_max_threads() * 16;
      serialized_partition_nbytes_scan = new size_t[serialized_npartitions+1];
      serialized_partition_nnz_scan = new size_t[serialized_npartitions+1];
    }

    buffer() : buffer(0,0) {}

    void alloc_serialized(size_t sz)
    {
      delete [] serialized_data;
      serialized_data = new char[sz]; 
      serialized_nbytes = sz;
    }

    int compute_nnz() const
    {
      int len = 0;
      #pragma omp parallel for reduction(+:len)
      for (int ii = 0 ; ii < num_ints ; ii++) {
        int p = _popcnt32(bit_vector[ii]);
        len += p;
      }
      return len;
    }
  
    int compute_nnz(int start, int finish) const
    {
      int len = 0;
      #pragma omp parallel for reduction(+:len)
      for (int ii = start  ; ii < finish ; ii++) {
        int p = _popcnt32(bit_vector[ii]);
        len += p;
      }
      return len;
    }  

    template<bool EXTENDS_SERIALIZABLE = std::is_base_of<Serializable,T>::value, 
             typename std::enable_if<EXTENDS_SERIALIZABLE>::type* = nullptr>
    void decompress()
    {
      memset(bit_vector, 0, num_ints* sizeof(int));
      std::stringstream * sss = new std::stringstream[serialized_npartitions];
      #pragma omp parallel for
      for(int p = 0 ; p < serialized_npartitions ; p++)
      {
        int i_per_partition = (num_ints + serialized_npartitions - 1) / serialized_npartitions;
        int start_i = i_per_partition * p;
        int end_i = i_per_partition * (p+1);
        if(end_i > num_ints) end_i = num_ints;
        sss[p].write(serialized_data + serialized_partition_nbytes_scan[p], 
                     (serialized_partition_nbytes_scan[p+1]-serialized_partition_nbytes_scan[p]));
        boost::archive::binary_iarchive ia(sss[p]);
        for(unsigned long int i = 0 ; i < (serialized_partition_nnz_scan[p+1] - 
                                           serialized_partition_nnz_scan[p]) ; i++)
        {
          int idx;
          ia >> idx;
          ia >> value[idx];
          set_bitvector(idx, bit_vector);
        }
      }
      delete [] sss;
    }

    template<bool EXTENDS_SERIALIZABLE = std::is_base_of<Serializable,T>::value, 
             typename std::enable_if<!EXTENDS_SERIALIZABLE>::type* = nullptr>
    void decompress()
    {
      memset(bit_vector, 0, num_ints* sizeof(int));
      //compressed_indices = reinterpret_cast<int*>(compressed_data + nnz);
      int npartitions = omp_get_max_threads();
      int * start_nnzs = new int[npartitions];
      int * end_nnzs = new int[npartitions];
      int mystart = 0;
      int my_nz_per = (nnz + npartitions - 1) / npartitions;
      my_nz_per = ((my_nz_per + 31) / 32) * 32;
      for(int p = 0 ; p < npartitions ; p++)
      {
        start_nnzs[p] = mystart;
        mystart += my_nz_per;
        if(mystart > nnz) mystart = nnz;
        if(mystart < nnz)
        {
          int start32 = compressed_indices[mystart] / 32;
          while((mystart < nnz) && compressed_indices[mystart] / 32  == start32) mystart++;
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
          int idx = compressed_indices[i];
          set_bitvector(idx, bit_vector);
          value[idx] = compressed_data[i];
        }
      }
      delete [] start_nnzs;
      delete [] end_nnzs;
    }

    template<bool EXTENDS_SERIALIZABLE = std::is_base_of<Serializable,T>::value, 
             typename std::enable_if<EXTENDS_SERIALIZABLE>::type* = nullptr>
    void compress()
    {
      size_t * serialized_partition_nbytes = new size_t[serialized_npartitions];
      size_t * serialized_partition_nnz = new size_t[serialized_npartitions];
      std::stringstream * sss = new std::stringstream[serialized_npartitions];

      #pragma omp parallel for
      for(int p = 0 ; p < serialized_npartitions ; p++)
      {
        int i_per_partition = (num_ints + serialized_npartitions - 1) / serialized_npartitions;
        int start_i = i_per_partition * p;
        int end_i = i_per_partition * (p+1);
        if(end_i > num_ints) end_i = num_ints;
        serialized_partition_nnz[p] = 0;
        boost::archive::binary_oarchive oa(sss[p]);
        
        for(int ii = start_i ; ii < end_i ; ii++)
        {
          if(_popcnt32(bit_vector[ii]) == 0) continue;
          for(int i = ii*32 ; i < (ii+1)*32 ; i++)
          {
            if(get_bitvector(i, bit_vector))
            {
              oa << i;
              oa << value[i];
              serialized_partition_nnz[p]++;
            }
          }
        }

        sss[p].seekg(0, sss[p].end);
        size_t sz = sss[p].tellg();
        sss[p].seekg(0, sss[p].beg);
        serialized_partition_nbytes[p] = sz;
      }

      serialized_partition_nnz_scan[0] = 0;
      serialized_partition_nbytes_scan[0] = 0;
      for(int p = 0 ; p < serialized_npartitions ; p++)
      {
        serialized_partition_nnz_scan[p+1] = serialized_partition_nnz_scan[p] + serialized_partition_nnz[p];
        serialized_partition_nbytes_scan[p+1] = serialized_partition_nbytes_scan[p] + serialized_partition_nbytes[p];
      }
      size_t sz = serialized_partition_nbytes_scan[serialized_npartitions];

      alloc_serialized(sz);

      #pragma omp parallel for
      for(int p = 0 ; p < serialized_npartitions ; p++)
      {
        sss[p].read(serialized_data + serialized_partition_nbytes_scan[p], serialized_partition_nbytes[p]);
      }

      delete [] serialized_partition_nnz;
      delete [] serialized_partition_nbytes;
      delete [] sss;

    }
  
    template<bool EXTENDS_SERIALIZABLE = std::is_base_of<Serializable,T>::value, 
             typename std::enable_if<!EXTENDS_SERIALIZABLE>::type* = nullptr>
    void compress()
    {
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
      nnz = 0;
      for(int p = 0 ; p < npartitions ; p++)
      {
        partition_nnz_scan[p+1] = partition_nnz_scan[p] + partition_nnz[p];
        nnz += partition_nnz[p];
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
          if(_popcnt32(bit_vector[ii]) == 0) continue;
          for(int i = ii*32 ; i < (ii+1)*32 ; i++)
          {
            if(get_bitvector(i, bit_vector))
            {
              compressed_data[nzcnt] = value[i];
              compressed_indices[nzcnt] = i;
              nzcnt++;
            }
          }
        }
      }
      delete [] partition_nnz;
      delete [] partition_nnz_scan;
    }

    ~buffer()
    {
      delete [] value;
      delete [] bit_vector;
      delete [] compressed_data;
      delete [] compressed_indices;
      delete [] serialized_partition_nbytes_scan;
      delete [] serialized_partition_nnz_scan;
      delete [] serialized_data;
    }
};

template <typename T>
class DenseSegment {
 public:
  std::string name;
  int capacity;
  int num_ints;

  buffer<T> *properties;
  send_metadata received_md;
  std::vector<send_metadata> queued_md;
  std::vector<buffer<T> * > received;
  std::vector<buffer<T> * > uninitialized;

  friend boost::serialization::access;
  template<class Archive> 
  void save(Archive& ar, const unsigned int version) const {
    bool properties_is_null = (properties == NULL);
    ar & properties_is_null;
    ar & name;
    ar & capacity;
    ar & num_ints;
    if(properties != NULL)
    {
      ar & properties;
    }
    ar & received_md;
    ar & queued_md;
    ar & received;
    ar & uninitialized;
  }

  template<class Archive> 
  void load(Archive& ar, const unsigned int version) {
    bool properties_null;
    ar & properties_null;
    ar & name;
    ar & capacity;
    ar & num_ints;
    if(!properties_null)
    {
      ar & properties;
    }
    else
    {
      properties = NULL;
    }
    ar & received_md;
    ar & queued_md;
    ar & received;
    ar & uninitialized;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

  DenseSegment(int n) {
    capacity = n;
    num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
    properties = NULL;
  }

  DenseSegment() : DenseSegment(0) {} 

  void ingestEdges(edge_t<T>* edges, int _m, int _nnz, int row_start)
  {
    alloc();
    initialize();

    for (uint64_t i = 0; i < (uint64_t)_nnz; i++) {
      int src = edges[i].src - row_start - 1;
      set_bitvector(src, properties->bit_vector);
      properties->value[src] = edges[i].val;
    }
    properties->nnz = _nnz;
    properties->uninitialized = false;
  }

  ~DenseSegment() 
  {
    if(properties != NULL)
    {
      delete properties;
    }

    for(auto it = received.begin() ; it != received.end() ; it++)
    {
      delete *it;
    }
    received.clear();

    for(auto it = uninitialized.begin() ; it != uninitialized.end() ; it++)
    {
      delete *it;
    }
    uninitialized.clear();
  }

  int compute_nnz() const
  {
    if(properties == NULL) return 0;
    if(properties->uninitialized) return 0;
    return properties->compute_nnz();
  }
  
  int compute_nnz(int start, int finish) const
  {
    if(properties == NULL) return 0;
    if(properties->uninitialized) return 0;
    return properties->compute_nnz(start, finish);
  }

  compression_decision should_compress(int test_nnz)
  {
    if(std::is_base_of<Serializable,T>::value) return SERIALIZED;
    if(test_nnz > get_compression_threshold() * capacity) 
      return NONE;
    return COMPRESSED;
  }
  void compress()
  {
    alloc();
    initialize();
    if(should_compress(properties->nnz) == COMPRESSED ||
       should_compress(properties->nnz) == SERIALIZED)
    {
      properties->compress();
    }
  }
  void decompress()
  {
    assert(properties);
    if(should_compress(properties->nnz) == COMPRESSED ||
       should_compress(properties->nnz) == SERIALIZED)
    {
      properties->decompress();
    }
  }

  void set_uninitialized_received()
  {
    for(auto it = received.begin() ; it != received.end() ; it++)
    {
      (*it)->uninitialized = true;
      uninitialized.push_back(*it);
    }
    received.clear();
  }

  void set_uninitialized() {
    set_uninitialized_received();
    if(properties != NULL)
    { 
      properties->uninitialized = true;
      properties->nnz = 0;
    }
  }

  void alloc() {
    if(properties == NULL)
    {
      properties = new buffer<T>(capacity, num_ints);
    }
  }

  void initialize()
  {
    if(properties->uninitialized)
    {
      memset(properties->bit_vector, 0, num_ints* sizeof(int));
      properties->nnz = 0;
    }
    properties->uninitialized = false;
  }

  int getNNZ()
  {
    return properties->nnz;
  }

  void set(int idx, T val) {
    alloc();
    initialize();
    if(!get_bitvector(idx-1, properties->bit_vector)) properties->nnz++;
    properties->value[idx - 1] = val;
    set_bitvector(idx-1, properties->bit_vector);
    properties->uninitialized = false;
  }

  void unset(int idx) {
    alloc();
    initialize();
    if(get_bitvector(idx-1, properties->bit_vector)) properties->nnz--;
    clear_bitvector(idx-1, properties->bit_vector);
    properties->uninitialized = false;
  }

  void setAll(T val) {
    alloc();
    //initialize();
    properties->uninitialized=false;
    if(num_ints == 0) return;
    properties->bit_vector[num_ints-1] = 0;
    #pragma omp parallel for
    for(int i = 0 ; i < num_ints-1 ; i++)
    {
      properties->bit_vector[i] = 0xFFFFFFFF;
    }

    for(int idx = std::max(0, capacity-32) ; idx < capacity ; idx++)
    {
      set_bitvector(idx, properties->bit_vector);
    }
    properties->nnz = capacity;

    #pragma omp parallel for
    for(int i = 0 ; i < capacity ; i++)
    {
      properties->value[i] = val;
    }
  }

  T get(const int idx) const {
    assert(properties);
    assert(!properties->uninitialized);
    return properties->value[idx - 1];
  }

  void send_nnz(int myrank, int dst_rank, std::vector<MPI_Request>* requests) {
    send_metadata md = {properties->nnz, properties->serialized_nbytes, properties->serialized_npartitions};
    MPI_Send(&md, sizeof(md), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD);
  }
  void recv_nnz_queue(int myrank, int src_rank,
                 std::vector<MPI_Request>* requests) {
    send_metadata md;
    MPI_Recv(&md, sizeof(md), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    queued_md.insert(queued_md.begin(), md);
  }
  void recv_nnz(int myrank, int src_rank,
                std::vector<MPI_Request>* requests) {
   alloc();
   MPI_Recv(&received_md, sizeof(send_metadata), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
  }
  void send_segment(int myrank, int dst_rank, std::vector<MPI_Request>* requests) {
    if(should_compress(properties->nnz) == COMPRESSED)
    {
      MPI_Request r1;
      MPI_Request r2;
      MPI_Isend(properties->compressed_data, properties->nnz * sizeof(T), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD,
                 &r1);
      MPI_Isend(properties->compressed_indices, properties->nnz * sizeof(int), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD,
                 &r2);
      requests->push_back(r1);
      requests->push_back(r2);
    }
    else if(should_compress(properties->nnz) == SERIALIZED)
    {
      MPI_Request r1;
      MPI_Request r2;
      MPI_Request r3;
      MPI_Isend(properties->serialized_data, properties->serialized_nbytes, MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD, &r1);
      MPI_Isend(properties->serialized_partition_nnz_scan, (properties->serialized_npartitions+1) * sizeof(size_t), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD, &r2);
      MPI_Isend(properties->serialized_partition_nbytes_scan, (properties->serialized_npartitions+1) * sizeof(size_t), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD, &r3);
      requests->push_back(r1);
      requests->push_back(r2);
      requests->push_back(r3);
    }
    else
    {
      MPI_Request r1;
      MPI_Request r2;
      MPI_Isend(properties->value, capacity * sizeof(T), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD,
                 &r1);
      MPI_Isend(properties->bit_vector, num_ints * sizeof(int), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD,
                 &r2);
      requests->push_back(r1);
      requests->push_back(r2);
    }
  }

  void recv_buffer(send_metadata md,
                   buffer<T> * p,
                   int myrank, int src_rank, 
                 std::vector<MPI_Request>* requests) {

    p->nnz = md.nnz;
    p->serialized_nbytes = md.serialized_nbytes;
    p->serialized_npartitions = md.serialized_npartitions;
    if(should_compress(p->nnz) == COMPRESSED)
    {
      MPI_Request r1;
      MPI_Request r2;
      MPI_Irecv(p->compressed_data, p->nnz * sizeof(T), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r1);
      MPI_Irecv(p->compressed_indices, p->nnz * sizeof(int), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r2);
      requests->push_back(r1);
      requests->push_back(r2);
    }
    else if(should_compress(p->nnz) == SERIALIZED)
    {
      MPI_Request r1;
      MPI_Request r2;
      MPI_Request r3;
      p->alloc_serialized(p->serialized_nbytes);
      MPI_Irecv(p->serialized_data, p->serialized_nbytes, MPI_BYTE, src_rank, 0, MPI_COMM_WORLD, &r1);
      MPI_Irecv(p->serialized_partition_nnz_scan, (p->serialized_npartitions+1) * sizeof(size_t), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD, &r2);
      MPI_Irecv(p->serialized_partition_nbytes_scan, (p->serialized_npartitions+1) * sizeof(size_t), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD, &r3);
      requests->push_back(r1);
      requests->push_back(r2);
      requests->push_back(r3);
    }
    else
    {
      MPI_Request r1;
      MPI_Request r2;
      MPI_Irecv(p->value, capacity * sizeof(T), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r1);
      MPI_Irecv(p->bit_vector, num_ints* sizeof(int), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r2);
      requests->push_back(r1);
      requests->push_back(r2);
    }
    p->uninitialized = false;
  }

  void recv_segment_queue(int myrank, int src_rank, 
                 std::vector<MPI_Request>* requests) {

    buffer<T> * new_properties;
    if(uninitialized.size() > 0) 
    {
      new_properties = uninitialized.back();
      uninitialized.pop_back();
    }
    else
    {
      new_properties = new buffer<T>(capacity, num_ints);
    }
    send_metadata md = queued_md.back();
    queued_md.pop_back();

    recv_buffer(md, new_properties, myrank, src_rank, requests);
    received.push_back(new_properties);
  }

  void recv_segment(int myrank, int src_rank, 
                 std::vector<MPI_Request>* requests) {

    recv_buffer(received_md, properties, myrank, src_rank, requests);
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
      if(get_bitvector(i, properties->bit_vector))
      {
        fout << i + start_id << " " << properties->value[i] << std::endl;
      }
    }
    fout.close();
  }

  void get_edges(edge_t<T> * edges, unsigned int start_nz) const
  {
    unsigned int mycnt = 0;
    for(int i = 0 ; i < capacity ; i++)
    {
      if(get_bitvector(i, properties->bit_vector))
      {
        edges[mycnt].src = start_nz + i + 1;
        edges[mycnt].dst = 1;
        edges[mycnt].val = properties->value[i];
        mycnt++;
      }
    }
  }

  template <typename Ta, typename Tb, typename Tc>
  void union_received(void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {
    alloc();
    initialize();
    for(auto it = received.begin() ; it != received.end() ; it++)
    {
      if(should_compress((*it)->nnz) == COMPRESSED)
      {
        union_compressed((*it)->compressed_data, (*it)->compressed_indices, (*it)->nnz, capacity, num_ints, properties->value, properties->bit_vector, op_fp, vsp);
      }
      else if(should_compress((*it)->nnz) == SERIALIZED)
      {
        (*it)->decompress();
        union_dense((*it)->value, (*it)->bit_vector, capacity, num_ints, properties->value, properties->bit_vector, properties->value, properties->bit_vector, op_fp, vsp);
      }
      else
      {
        union_dense((*it)->value, (*it)->bit_vector, capacity, num_ints, properties->value, properties->bit_vector, properties->value, properties->bit_vector, op_fp, vsp);
      }
    } 
  }
};


#endif  // SRC_DENSESEGMENT_H_
