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


#ifndef SRC_TILEOPS_H_
#define SRC_TILEOPS_H_

#include "src/HybridTile.h"
#include "src/CSRTile.h"
#include "src/COOTile.h"
#include "src/COOSIMD32Tile.h"
#include "src/DCSCTile.h"
#include "src/DenseTile.h"
#include "src/DenseSegment.h"
#include "src/singlenode_fspgemm.h"
#include "src/singlenode_fspmspv.h"
#include "src/singlenode_intersectreduce.h"
#include "src/singlenode_reduce.h"
#include "src/singlenode_spgemm.h"
#include "src/singlenode_spmspv.h"
#include "src/singlenode_spmspv3.h"
#include "src/singlenode_unionreduce.h"
#include "src/singlenode_apply.h"
#include "src/singlenode_sparsify.h"
#include "src/singlenode_clear.h"

// CSR
template <typename Ta, typename Tb, typename Tc>
void mult_tile(const CSRTile<Ta>& tilea, const CSRTile<Tb>& tileb,
               CSRTile<Tc>* tilec_in, int myrank, int output_rank,
               void (*mul_fp)(Ta, Tb, Tc*, void*), void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
  Tc* ac;
  int* jc;
  int* ic;

  if (!(tilea.isEmpty()) && !(tileb.isEmpty()) && !(tilec_in->isEmpty())) {
    my_dcsrmultcsr(tilea.m, tileb.n, tileb.m, tilea.a, tilea.ja, tilea.ia,
                   tileb.a, tileb.ja, tileb.ia, &ac, &jc, &ic, tilec_in->a,
                   tilec_in->ja, tilec_in->ia, mul_fp, add_fp, vsp);
    tilec_in->clear();
    tilec_in->nnz = ic[tilec_in->m] - 1;
    tilec_in->a = ac;
    tilec_in->ia = ic;
    tilec_in->ja = jc;
  } else if (!(tilea.isEmpty()) && !(tileb.isEmpty()) &&
             (tilec_in->isEmpty())) {
    my_dcsrmultcsr(tilea.m, tileb.n, tileb.m, tilea.a, tilea.ja, tilea.ia,
                   tileb.a, tileb.ja, tileb.ia, &ac, &jc, &ic,
                   reinterpret_cast<Tc*>(NULL), reinterpret_cast<int*>(NULL),
                   reinterpret_cast<int*>(NULL), mul_fp, add_fp, vsp);
    tilec_in->clear();
    tilec_in->nnz = ic[tilec_in->m] - 1;
    tilec_in->a = ac;
    tilec_in->ia = ic;
    tilec_in->ja = jc;
  }
}

template <typename Ta, typename Tb, typename Tc, typename Tf>
void fmult_tile(const CSRTile<Ta>& tilea, const CSRTile<Tb>& tileb,
                CSRTile<Tc>* tilec_in, const CSRTile<Tf>& tilef, int myrank,
                int output_rank, void (*mul_fp)(Ta, Tb, Tc*, void*), void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
  Tc* ac;
  int* jc;
  int* ic;

  if (!(tilea.isEmpty()) && !(tileb.isEmpty()) && !(tilec_in->isEmpty())) {
    my_fcsrmultcsr(tilea.m, tileb.n, tileb.m, tilea.a, tilea.ja, tilea.ia,
                   tileb.a, tileb.ja, tileb.ia, &ac, &jc, &ic, tilec_in->a,
                   tilec_in->ja, tilec_in->ia, tilef.a, tilef.ja, tilef.ia,
                   mul_fp, add_fp, vsp);
    tilec_in->clear();
    tilec_in->nnz = ic[tilec_in->m] - 1;
    tilec_in->a = ac;
    tilec_in->ia = ic;
    tilec_in->ja = jc;
  } else if (!(tilea.isEmpty()) && !(tileb.isEmpty()) &&
             (tilec_in->isEmpty())) {
    my_fcsrmultcsr(tilea.m, tileb.n, tileb.m, tilea.a, tilea.ja, tilea.ia,
                   tileb.a, tileb.ja, tileb.ia, &ac, &jc, &ic, reinterpret_cast<Tc*>(NULL), reinterpret_cast<int*>(NULL), reinterpret_cast<int*>(NULL),
                   tilef.a, tilef.ja, tilef.ia, mul_fp, add_fp, vsp);
    tilec_in->clear();
    tilec_in->nnz = ic[tilec_in->m] - 1;
    tilec_in->a = ac;
    tilec_in->ia = ic;
    tilec_in->ja = jc;
  }
}

template <typename T>
void union_tile(const CSRTile<T>& tilea, CSRTile<T>* tileb, int myrank,
                int output_rank, T (*add_fp)(T, T)) {
  T* ac;
  int* jc;
  int* ic;

  if (!(tilea.isEmpty()) && !(tileb->isEmpty())) {
    my_dcsradd(tilea.m, tilea.n, tilea.a, tilea.ja, tilea.ia, tileb->a,
               tileb->ja, tileb->ia, &ac, &jc, &ic, add_fp);
    tileb->clear();
    tileb->nnz = ic[tileb->m] - 1;
    tileb->a = ac;
    tileb->ia = ic;
    tileb->ja = jc;
  } else if ((tileb->isEmpty()) && !(tilea.isEmpty())) {
    tileb->nnz = tilea.nnz;
    tileb->a = reinterpret_cast<T*>(
        _mm_malloc((uint64_t)tilea.nnz * (uint64_t)sizeof(T), 64));
    tileb->ia =
        reinterpret_cast<int*>(_mm_malloc((tilea.m + 1) * sizeof(int), 64));
    tileb->ja = reinterpret_cast<int*>(
        _mm_malloc((uint64_t)tilea.nnz * (uint64_t)sizeof(int), 64));
    memcpy(tileb->a, tilea.a, tilea.nnz * sizeof(T));
    memcpy(tileb->ia, tilea.ia, (tilea.m + 1) * sizeof(T));
    memcpy(tileb->ja, tilea.ja, tilea.nnz * sizeof(T));
  }
}

template <typename Ta, typename Tb, typename Tc>
void intersect_tile(const CSRTile<Ta>& tilea, const CSRTile<Tb>& tileb,
                    CSRTile<Tc>* tilec, int myrank, int output_rank,
                    void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {
  Tc* ac;
  int* jc;
  int* ic;

  if (!(tilea.isEmpty()) && !(tileb.isEmpty())) {
    my_dintersect(tilea.m, tilea.n, tilea.a, tilea.ja, tilea.ia, tileb.a,
                  tileb.ja, tileb.ia, &ac, &jc, &ic, op_fp, vsp);
    tilec->clear();
    tilec->nnz = ic[tilec->m] - 1;
    tilec->a = ac;
    tilec->ia = ic;
    tilec->ja = jc;
  }
}

template <typename T>
void reduce_tile(const CSRTile<T>& tile, T* result, bool* res_set,
                 void (*op_fp)(T, T, T*, void*), void* vsp) {
  reduce_csr(tile.a, tile.nnz, result, res_set, op_fp, vsp);
}

template <typename T>
void reduce_tile(const DCSCTile<T>& tile, T* result, bool* res_set,
                 void (*op_fp)(T, T, T*, void*), void* vsp) {
  reduce_dcsc(tile.vals, tile.nnz, result, res_set, op_fp, vsp);
}

template <typename T>
void reduce_tile(const DCSCTile<T>& tile, DenseSegment<T> * result,
                 void (*op_fp)(T, T, T*, void*), void* vsp) {
    result->alloc();
    result->initialize();
    reduce_dcsc(tile.row_inds, tile.col_ptrs, tile.col_indices,
                     tile.vals, tile.num_partitions, tile.row_pointers,
                     tile.col_starts, tilea.edge_pointers, result->properties.value,
                     result->properties.bit_vector, op_fp, vsp);
}

template <typename Ta, typename Tx, typename Ty>
void mult_segment(const DCSCTile<Ta>& tile, DenseSegment<Tx>& segmentx,
                  DenseSegment<Ty>* segmenty, int output_rank,
                  void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp) {
  segmenty->alloc();
  segmenty->initialize();
  my_spmspv(tile.row_inds, tile.col_ptrs, tile.col_indices, tile.vals,
            tile.num_partitions, tile.row_pointers, tile.col_starts,
            tile.edge_pointers, segmentx.properties.value, segmentx.properties.bit_vector,
            segmenty->properties.value, segmenty->properties.bit_vector, tile.m, tile.n, (&segmenty->properties.nnz),
            mul_fp, add_fp, vsp);
  segmenty->properties.nnz = segmenty->compute_nnz();
}

template <typename Ta, typename Tx, typename Tvp, typename Ty>
void mult_segment3(const DCSCTile<Ta>& tile, const DenseSegment<Tx>& segmentx,
                  const DenseSegment<Tvp> & segmentvp,
                  DenseSegment<Ty>* segmenty, int output_rank,
                  Ty (*mul_fp)(Ta, Tx, Tvp), Ty (*add_fp)(Ty, Ty)) {
  segmenty->alloc();
  int nnz = 0;
  my_spmspv3(tile.row_inds, tile.col_ptrs, tile.col_indices, tile.vals,
            tile.num_partitions, tile.row_pointers, tile.col_starts,
            tile.edge_pointers, segmentx.properties.value, segmentx.properties.bit_vector,
	    segmentvp.properties.value, segmentvp.properties.bit_vector,
            segmenty->properties.value, segmenty->properties.bit_vector, tile.m, tile.n, (&nnz),
            mul_fp, add_fp);
}

template <typename Ta, typename Tx, typename Ty>
void mult_segment(const HybridTile<Ta>& tile, const DenseSegment<Tx>& segmentx,
                  DenseSegment<Ty>* segmenty, int output_rank,
                  void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp) {
  segmenty->alloc();
  segmenty->initialize();
  int nnz = 0;
  if(tile.t1->nnz > 0)
  {
    my_dcsrspmspv(tile.t1->a, tile.t1->ia, tile.t1->ja, tile.t1->row_ids, tile.t1->num_rows, tile.t1->partition_ptrs, tile.t1->num_partitions, segmentx.properties.value, segmentx.properties.bit_vector,
                 segmenty->properties.value, segmenty->properties.bit_vector, tile.t1->m, tile.t1->n, (&nnz),
                 mul_fp, add_fp, vsp);
     /*
    my_csrspmspv(tile.t1->a, tile.t1->ia, tile.t1->ja, segmentx.properties.value, segmentx.properties.bit_vector,
                 segmenty->properties.value, segmenty->properties.bit_vector, tile.t1->m, tile.t1->n, (&nnz),
                 mul_fp, add_fp, vsp);
                 */
  }
  if(tile.t2->nnz > 0)
  {
    my_coospmspv(tile.t2->a, tile.t2->ia, tile.t2->ja, tile.t2->num_partitions, tile.t2->partition_start,
                 segmentx.properties.value, segmentx.properties.bit_vector,
                 segmenty->properties.value, segmenty->properties.bit_vector, tile.t2->m, tile.t2->n, (&nnz),
                 mul_fp, add_fp, vsp);
  }
  segmenty->properties.nnz = segmenty->compute_nnz();
}


template <typename Ta, typename Tx, typename Ty>
void mult_segment(const CSRTile<Ta>& tile, const DenseSegment<Tx>& segmentx,
                  DenseSegment<Ty>* segmenty, int output_rank,
                  void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp) {
  segmenty->alloc();
  segmenty->initialize();
  int nnz = 0;
  if(tile.nnz > 0)
  {
    my_csrspmspv(tile.a, tile.ia, tile.ja, segmentx.properties.value, segmentx.properties.bit_vector,
                 segmenty->properties.value, segmenty->properties.bit_vector, tile.m, tile.n, (&nnz),
                 mul_fp, add_fp, vsp);
  }
  segmenty->properties.nnz = segmenty->compute_nnz();
}


template <typename Ta, typename Tx, typename Ty>
void mult_segment(const COOTile<Ta>& tile, const DenseSegment<Tx>& segmentx,
                  DenseSegment<Ty>* segmenty, int output_rank,
                  void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp) {
  segmenty->alloc();
  segmenty->initialize();
  int nnz = 0;
  my_coospmspv(tile.a, tile.ia, tile.ja, tile.num_partitions, tile.partition_start,
               segmentx.properties.value, segmentx.properties.bit_vector,
               segmenty->properties.value, segmenty->properties.bit_vector, tile.m, tile.n, (&nnz),
               mul_fp, add_fp, vsp);
  segmenty->properties.nnz = segmenty->compute_nnz();
}

template <typename Ta, typename Tx, typename Ty>
void mult_segment(const COOSIMD32Tile<Ta>& tile, const DenseSegment<Tx>& segmentx,
                  DenseSegment<Ty>* segmenty, int output_rank,
                  void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp) {
  segmenty->alloc();
  segmenty->initialize();
  int nnz = 0;
  my_coospmspv(tile.a, tile.ia, tile.ja, tile.num_partitions, tile.partition_start,
               segmentx.properties.value, segmentx.properties.bit_vector,
               segmenty->properties.value, segmenty->properties.bit_vector, tile.m, tile.n, (&nnz),
               mul_fp, add_fp, vsp);
  segmenty->properties.nnz = segmenty->compute_nnz();
}



template <typename Ta, typename Tx, typename Ty, typename Tm>
void fmult_segment(const CSRTile<Ta>& tile, const DenseSegment<Tx>& segmentx, DenseSegment<Tm> &segmentm,
                  DenseSegment<Ty>* segmenty, int output_rank,
                  void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp) {
  //segmenty->alloc();
  //segmenty->initialize(); // Assume must be done
  my_csrfspmspv(tile.a, tile.ia, tile.ja, segmentx.properties.value, segmentx.properties.bit_vector,
               segmentm.properties.bit_vector,
               segmenty->properties.value, segmenty->properties.bit_vector, tile.m, tile.n,
               mul_fp, add_fp, vsp);
  //segmenty->properties.nnz = segmenty->compute_nnz();
}


template <typename Ta, typename Tb, typename Tc>
void mult_tile(const DCSCTile<Ta>& tilea, const DenseTile<Tb>& tileb,
               DenseTile<Tc>* tilec_in, int myrank, int output_rank,
               void (*mul_fp)(Ta, Tb, Tc*, void*), void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
  // If C is empty then create it
  if (tilec_in->isEmpty()) {
    tilec_in->alloc();
  }

  if (!tilea.isEmpty() && !tileb.isEmpty()) {
    my_dcscmultdense(tilea.row_inds, tilea.col_ptrs, tilea.col_indices,
                     tilea.vals, tilea.num_partitions, tilea.row_pointers,
                     tilea.col_starts, tilea.edge_pointers, tileb.value,
                     tileb.bit_vector, tilec_in->value, tilec_in->bit_vector,
                     tilea.m, tileb.n, tilea.n, &(tilec_in->nnz), mul_fp,
                     add_fp, vsp);
  }
}

template <typename T>
void union_tile(const DenseTile<T>& tilea, DenseTile<T>* tileb, int myrank,
                int output_rank, T (*add_fp)(T, T)) {
  // If C is empty then create it
  if (tileb->isEmpty()) {
    tileb->alloc();
  }

  if (!tilea.isEmpty()) {
    my_denseadd(tilea.value, tilea.bit_vector, tileb->value, tileb->bit_vector,
                tilea.m, tilea.n, &(tileb->nnz), add_fp);
  }
}

template <typename T>
void reduce_tile(const DenseTile<T>& tile, T* result, bool* res_set,
                 T (*op_fp)(T, T)) {
  reduce_dense(tile.value, tile.bit_vector, tile.m, tile.n, result, res_set,
               op_fp);
}

template <typename T>
void reduce_tile(const DenseTile<T>& tile, DenseSegment<T> * result,
                 void (*op_fp)(T, T, T*, void*), void* vsp) {

  result->alloc();
  result->initialize();
  reduce_dense(tile.value, tile.bit_vector, tile.m, tile.n, result->properties.value, result->properties.bit_vector, 
               op_fp, vsp);
}


template <typename T>
void reduce_segment(const DenseSegment<T>& segment, T* res, bool* res_set,
                    void (*op_fp)(T, T, T*, void*), void* vsp) {

  reduce_dense_segment(segment.properties.value, segment.properties.bit_vector, segment.capacity, res, res_set, op_fp, vsp);
}

template <typename VT, typename T>
void mapreduce_segment(DenseSegment<VT> * segment, T* res, bool* res_set,
                    void (*op_map)(VT*, T*, void*), void (*op_fp)(T, T, T*, void*), void* vsp) {

  mapreduce_dense_segment(segment->properties.value, segment->properties.bit_vector, segment->capacity, res, res_set, op_map, op_fp, vsp);
}



template <typename Ta, typename Tb>
void apply_segment(const DenseSegment<Ta> & s_in, DenseSegment<Tb> * s_out, 
                    void (*add_fp)(Ta, Tb*, void*), void* vsp) {
  s_out->alloc();
  s_out->initialize();
  apply_dense_segment(s_in.properties.value, s_in.properties.bit_vector, &(s_out->properties.nnz), s_in.num_ints, s_out->properties.value, s_out->properties.bit_vector,  add_fp, vsp);
}

template <typename Ta, typename Tb>
void sparsify_segment(const DenseSegment<Ta> & s_in, DenseSegment<Tb> * s_out, 
                    void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp) {
  s_out->alloc();
  s_out->initialize();
  sparsify_dense_segment(s_in.properties.value, s_in.properties.bit_vector, s_in.capacity, s_in.num_ints, s_out->properties.value, s_out->properties.bit_vector, &(s_out->properties.nnz), add_fp, vsp);
}

template <typename Ta, typename Tb>
void sparsify_tile(const DenseTile<Ta> & s_in, DenseTile<Tb> * s_out, 
                    void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp) {
  if (s_out->isEmpty()) {
    s_out->alloc();
  }
  printf("sparsify_tile\n");
  sparsify_dense_tile(s_in.value, s_in.bit_vector, s_in.m, s_in.n, s_out->value, s_out->bit_vector, &(s_out->nnz), add_fp, vsp);
}

template <typename Ta, typename Tb>
void sparsify_tile(const DenseTile<Ta> & s_in, CSRTile<Tb> * s_out, 
                    void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp) {
  printf("sparsify_tile csr\n");

  Tb* ac;
  int* jc;
  int* ic;

  sparsify_dense_to_csr(s_in.value, s_in.bit_vector, s_in.m, s_in.n, &ac, &jc, &ic, add_fp, vsp);

  s_out->clear();
  s_out->nnz = ic[s_out->m] - 1;
  s_out->a = ac;
  s_out->ia = ic;
  s_out->ja = jc;
}


template <typename Ta, typename Tb>
void apply_tile(const DenseTile<Ta> & s_in, DenseTile<Tb> * s_out, 
                    void (*add_fp)(Ta, Tb*, void*), void* vsp) {
  if (s_out->isEmpty()) {
    s_out->alloc();
  }
  apply_dense_tile(s_in.value, s_in.bit_vector, s_in.m, s_in.n, s_in.num_ints, s_out->value, s_out->bit_vector, add_fp, vsp);
}

template <typename Ta, typename Tb>
void apply_tile(const CSRTile<Ta> & s_in, DenseTile<Tb> * s_out, 
                    void (*add_fp)(Ta, Tb*, void*), void* vsp) {
  if (s_out->isEmpty()) {
    s_out->alloc();
  }
  apply_csr_to_dense(s_in.a, s_in.ja, s_in.ia, s_in.m, s_in.n, s_in.nnz, s_out->value, s_out->bit_vector, add_fp, vsp);
}



template <typename T>
void clear_segment(DenseSegment<T> * s1) {
  //s1->alloc();
  //clear_dense_segment(s1->properties.value, s1->properties.bit_vector, s1->num_ints);
  s1->set_uninitialized();
}

template <typename T>
void clear_tile(DenseTile<T> * s_out) {
  if (s_out->isEmpty()) {
    s_out->alloc();
  }

  clear_dense_tile(s_out->value, s_out->bit_vector, s_out->num_ints);
}


template <typename T>
void clear_tile(CSRTile<T> * s_out) {
  s_out->clear();
}

template <typename Ta, typename Tb, typename Tc>
void intersect_segment(const DenseSegment<Ta> & s1, const DenseSegment<Tb> & s2, DenseSegment<Tc> * s3, 
                    void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {

  s3->alloc();
  s3->initialize();
  if(!s1.properties.uninitialized && !s2.properties.uninitialized)
  {
    intersect_dense_segment(s1.properties.value, s1.properties.bit_vector, &(s3->properties.nnz), s1.num_ints, s2.properties.value, s2.properties.bit_vector, s3->properties.value, s3->properties.bit_vector, op_fp, vsp);
  }
}

template <typename Ta, typename Tb, typename Tc>
void union_segment(const DenseSegment<Ta> & s1, const DenseSegment<Tb> & s2, DenseSegment<Tc> * s3, 
                    void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {

  s3->alloc();
  union_dense_segment(s1.properties.value, s1.properties.bit_vector, s1.capacity, s1.num_ints, s2.properties.value, s2.properties.bit_vector, s3->properties.value, s3->properties.bit_vector, op_fp, vsp);
}

template <typename Ta, typename Tb, typename Tc>
void union_compress_segment(DenseSegment<Ta> * s1, 
                    void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {

  s1->alloc();
  s1->initialize();
  for(auto it = s1->received_properties.begin() ; it != s1->received_properties.end() ; it++)
  {
    assert(!(it->uninitialized));
    assert(it->allocated);
    if(s1->should_compress(it->nnz))
    {
      union_compressed_segment(it->compressed_data, it->nnz, s1->capacity, s1->num_ints, s1->properties.value, s1->properties.bit_vector, op_fp, vsp);
    }
    else
    {
      union_dense_segment(it->value, it->bit_vector, s1->capacity, s1->num_ints, s1->properties.value, s1->properties.bit_vector, s1->properties.value, s1->properties.bit_vector, op_fp, vsp);
    }
  }
}



#endif  // SRC_TILEOPS_H_
