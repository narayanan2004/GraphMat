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

#include "GMDP/matrices/edgelist.h"
#include "GMDP/utils/bitvector.h"
#include "GMDP/singlenode/unionreduce.h"

#include <string>
#include <vector>
#include <queue>
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
};

/* begin http://stackoverflow.com/questions/87372/check-if-a-class-has-a-member-function-of-a-given-signature */
template<typename T>
struct HasBoostSerializeMethod
{
    //template<typename U, void (U::*)(char*) const> struct SFINAE {};
    //template<typename U> static char Test(SFINAE<U, &U::serialize>*);
    template<typename U, void (U::*)(boost::archive::binary_iarchive&, const unsigned int) > struct SFINAE {};
    template<typename U> static char Test(SFINAE<U, &U::serialize >*);
    template<typename U> static int Test(...);
    static const bool Check = sizeof(Test<T>(0)) == sizeof(char);
};
/* end http://stackoverflow.com/questions/87372/check-if-a-class-has-a-member-function-of-a-given-signature */

template <typename T>
class buffer
{
  public:
    T * value;
    int * bit_vector;
    bool uninitialized;
    int nnz;
    size_t serialized_nbytes;
    int capacity;
    int num_ints;

    T * compressed_data;
    int * compressed_indices;

    char * serialized_data;

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
      serialized_data = NULL;
    }
    void clear_serialized()
    {
      if(serialized_data != NULL) _mm_free(serialized_data);
      serialized_data = NULL;
    }
    void alloc_serialized(size_t sz)
    {
      assert(serialized_data == NULL);
      serialized_data = reinterpret_cast<char*>(_mm_malloc(sz, 64));
    }

   /* begin http://stackoverflow.com/questions/16529512/c-class-member-function-specialization-on-bool-values  */
   template<bool M = HasBoostSerializeMethod<T>::Check, typename std::enable_if<M>::type* = nullptr>
   /* end http://stackoverflow.com/questions/16529512/c-class-member-function-specialization-on-bool-values  */
    void decompress()
    {
      std::stringstream ss;
      ss.write(serialized_data, serialized_nbytes);
      boost::archive::binary_iarchive ia(ss);
      memset(bit_vector, 0, num_ints* sizeof(int));
      for(unsigned long int i = 0 ; i < nnz ; i++)
      {
        int idx;
        ia >> idx;
        ia >> value[idx];
        set_bitvector(idx, bit_vector);
      }
    }

   /* begin http://stackoverflow.com/questions/16529512/c-class-member-function-specialization-on-bool-values  */
   template<bool M = HasBoostSerializeMethod<T>::Check, typename std::enable_if<!M>::type* = nullptr>
   /* end http://stackoverflow.com/questions/16529512/c-class-member-function-specialization-on-bool-values  */
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

    ~buffer()
    {
      delete [] value;
      delete [] bit_vector;
      //_mm_free(compressed_data);
      delete [] compressed_data;
      delete [] compressed_indices;
      clear_serialized();
    }
};

template <typename T>
class DenseSegment {
 public:
  std::string name;
  int capacity;
  int num_ints;

  buffer<T> *properties;
  std::queue<send_metadata> to_be_received;
  std::vector<buffer<T> * > received;
  std::vector<buffer<T> * > uninitialized;

  DenseSegment(int n) {
    capacity = n;
    num_ints = (n + sizeof(int) * 8 - 1) / (sizeof(int) * 8);
    properties = NULL;
  }

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
    int len = 0;
    #pragma omp parallel for reduction(+:len)
    for (int ii = 0 ; ii < num_ints ; ii++) {
      int p = _popcnt32(properties->bit_vector[ii]);
      len += p;
    }
    return len;
  }
  
  int compute_nnz(int start, int finish) const
  {
    if(properties == NULL) return 0;
    if(properties->uninitialized) return 0;
    int len = 0;
    #pragma omp parallel for reduction(+:len)
    for (int ii = start  ; ii < finish ; ii++) {
      int p = _popcnt32(properties->bit_vector[ii]);
      len += p;
    }
    return len;
  }

  compression_decision should_compress(int test_nnz)
  {
    if(HasBoostSerializeMethod<T>::Check) return SERIALIZED;
    if(test_nnz > get_compression_threshold() * capacity) 
      return NONE;
    return COMPRESSED;
  }

/* begin http://stackoverflow.com/questions/16529512/c-class-member-function-specialization-on-bool-values  */
  template<bool M = HasBoostSerializeMethod<T>::Check, typename std::enable_if<M>::type* = nullptr>
/* end http://stackoverflow.com/questions/16529512/c-class-member-function-specialization-on-bool-values  */
  void compress()
  {
    alloc();
    initialize();
    std::stringstream ss;
    int new_nnz = 0;
    {
    boost::archive::binary_oarchive oa(ss);

    for(int i = 0 ; i < capacity ; i++)
    {
      if(get_bitvector(i, properties->bit_vector))
      {
        //oa << i << properties->value[i];
        oa << i;
	oa << properties->value[i];
        new_nnz++;
      }
    }
    }
    
    ss.seekg(0, ss.end);
    size_t sz = ss.tellg();
    ss.seekg(0, ss.beg);
    properties->clear_serialized();
    properties->alloc_serialized(sz);
    ss.read(properties->serialized_data, sz);
    properties->nnz = new_nnz;
    properties->serialized_nbytes = sz;

  }

/* begin http://stackoverflow.com/questions/16529512/c-class-member-function-specialization-on-bool-values  */
  template<bool M = HasBoostSerializeMethod<T>::Check, typename std::enable_if<!M>::type* = nullptr>
/* end http://stackoverflow.com/questions/16529512/c-class-member-function-specialization-on-bool-values  */
  void compress()
  {
    alloc();
    initialize();
    if(should_compress(properties->nnz) == COMPRESSED)
    {
      //properties->compressed_indices = reinterpret_cast<int*>(properties->compressed_data + properties->nnz);
  
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
      properties->nnz = 0;
      for(int p = 0 ; p < npartitions ; p++)
      {
        partition_nnz_scan[p+1] = partition_nnz_scan[p] + partition_nnz[p];
        properties->nnz += partition_nnz[p];
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
          if(_popcnt32(properties->bit_vector[ii]) == 0) continue;
          for(int i = ii*32 ; i < (ii+1)*32 ; i++)
          {
            if(get_bitvector(i, properties->bit_vector))
            {
              properties->compressed_data[nzcnt] = properties->value[i];
              properties->compressed_indices[nzcnt] = i;
              nzcnt++;
            }
          }
        }
      }
      delete [] partition_nnz;
      delete [] partition_nnz_scan;
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
    send_metadata md = {properties->nnz, properties->serialized_nbytes};
    MPI_Send(&md, sizeof(md), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD);
  }
  void recv_nnz_queue(int myrank, int src_rank,
                 std::vector<MPI_Request>* requests) {
    send_metadata md;
    MPI_Recv(&md, sizeof(md), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    to_be_received.push(md);
  }
  void recv_nnz(int myrank, int src_rank,
                std::vector<MPI_Request>* requests) {
   alloc();
   send_metadata md;
   MPI_Recv(&md, sizeof(md), MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
            MPI_STATUS_IGNORE);
   properties->nnz = md.nnz;
   properties->serialized_nbytes = md.serialized_nbytes;
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
      MPI_Isend(properties->serialized_data, properties->serialized_nbytes, MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD,
                 &r1);
      requests->push_back(r1);
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

  void recv_buffer(buffer<T> * p,
                   int myrank, int src_rank, 
                 std::vector<MPI_Request>* requests) {

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
      p->clear_serialized();
      p->alloc_serialized(p->serialized_nbytes);
      MPI_Irecv(p->serialized_data, p->serialized_nbytes, MPI_BYTE, src_rank, 0, MPI_COMM_WORLD,
             &r1);
      requests->push_back(r1);
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
    send_metadata md = to_be_received.front();
    to_be_received.pop();
    new_properties->nnz = md.nnz;
    new_properties->serialized_nbytes = md.serialized_nbytes;

    recv_buffer(new_properties, myrank, src_rank, requests);
    received.push_back(new_properties);
  }

  void recv_segment(int myrank, int src_rank, 
                 std::vector<MPI_Request>* requests) {
    recv_buffer(properties, myrank, src_rank, requests);
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
