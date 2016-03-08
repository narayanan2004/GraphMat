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


#ifndef SRC_SINGLENODE_UNIONREDUCE_H_
#define SRC_SINGLENODE_UNIONREDUCE_H_

#include <algorithm>

#ifdef UNION_NAIVE_MERGE
template <typename T>
void my_dcsradd(int m, int n, T *a, int *ja, int *ia, T *b, int *jb, int *ib,
                T **c, int **jc, int **ic, T (*add_fp)(T, T)) {
#ifndef SORTED
#error Merge kernels require sorted inputs
#endif
  int nnzc = ia[m] - 1 + ib[m] - 1;
  int nzmax = nnzc;

  (*ic) = reinterpret_cast<int *>(_mm_malloc((m + 1) * sizeof(int), 64));
  (*c) = reinterpret_cast<T *>(
      _mm_malloc((uint64_t)nnzc * (uint64_t)sizeof(T), 64));
  (*jc) = reinterpret_cast<int *>(
      _mm_malloc((uint64_t)nnzc * (uint64_t)sizeof(int), 64));

  // Add new_c = c + tc
  int cnz_cnt = 0;
  for (int row = 0; row < m; row++) {
    *(ic)[row] = cnz_cnt + 1;

    // Merge c row and tc row into new_c row
    int Astart = ia[row];
    int Aend = ia[row + 1];
    int Bstart = ib[row];
    int Bend = ib[row + 1];

    while ((Astart < Aend) || (Bstart < Bend)) {
      int Acol = (Astart != Aend) ? ja[Astart - 1] : INT_MAX;
      int Bcol = (Bstart != Bend) ? jb[Bstart - 1] : INT_MAX;
      if (Acol < Bcol) {
        (*c)[cnz_cnt] = a[Astart - 1];
        (*jc)[cnz_cnt] = Acol;
        cnz_cnt++;
        Astart++;
      } else if (Bcol < Acol) {
        (*c)[cnz_cnt] = b[Bstart - 1];
        (*jc)[cnz_cnt] = Bcol;
        cnz_cnt++;
        Bstart++;
      } else {
        (*c)[cnz_cnt] = add_fp(a[Astart - 1], b[Bstart - 1]);
        (*jc)[cnz_cnt] = Acol;
        cnz_cnt++;
        Astart++;
        Bstart++;
#ifdef COUNT_FLOPS
        add_flops2++;
#endif
      }
    }
  }
  (*ic)[m] = cnz_cnt + 1;
}
#endif

#ifdef UNION_NAIVE_SPA

bool cmp_int_union_naive(int i1, int i2) { return i1 < i2; }

template <typename T>
void my_dcsradd(int m, int n, T *a, int *ja, int *ia, T *b, int *jb, int *ib,
                T **c, int **jc, int **ic, T (*add_fp)(T, T)) {
  T *Crow = reinterpret_cast<T *>(_mm_malloc(n * sizeof(T), 64));
  int *Cidxs = reinterpret_cast<int *>(_mm_malloc(n * sizeof(int), 64));
  bool *Cflags = reinterpret_cast<bool *>(_mm_malloc(n * sizeof(bool), 64));
  memset(Crow, 0, n * sizeof(T));
  memset(Cflags, 0, n * sizeof(bool));

  int nnzA = ia[m] - 1;
  int nnzB = ib[m] - 1;
  int nnzmax = nnzA + nnzB;

  (*ic) = reinterpret_cast<int *>(_mm_malloc((m + 1) * sizeof(int), 64));
  (*c) = reinterpret_cast<T *>(
      _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(T), 64));
  (*jc) = reinterpret_cast<int *>(
      _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(int), 64));

  int cnz_cnt = 0;
  for (int Arow = 0; Arow < m; Arow++) {
    int c_row_nz_start = cnz_cnt;
    (*ic)[Arow] = cnz_cnt + 1;

    for (int Anz_id = ia[Arow]; Anz_id < ia[Arow + 1]; Anz_id++) {
      int Acol = ja[Anz_id - 1];
      if (!Cflags[Acol - 1]) {
        (*jc)[cnz_cnt] = Acol;
        cnz_cnt++;
      }
      Cflags[Acol - 1] = true;
      Crow[Acol - 1] = add_fp(Crow[Acol - 1], a[Anz_id - 1]);
    }
    for (int Bnz_id = ib[Arow]; Bnz_id < ib[Arow + 1]; Bnz_id++) {
      int Bcol = jb[Bnz_id - 1];
      if (!Cflags[Bcol - 1]) {
        (*jc)[cnz_cnt] = Bcol;
        cnz_cnt++;
      }
      Cflags[Bcol - 1] = true;
      Crow[Bcol - 1] = add_fp(Crow[Bcol - 1], b[Bnz_id - 1]);
    }
#ifdef SORTED
    std::sort((*jc) + c_row_nz_start, (*jc) + cnz_cnt, cmp_int_union_naive);
#endif
    for (int Cnz_id = c_row_nz_start; Cnz_id < cnz_cnt; Cnz_id++) {
      int Ccol = (*jc)[Cnz_id];
      (*c)[Cnz_id] = Crow[Ccol - 1];
      Crow[Ccol - 1] = 0.0;
      Cflags[Ccol - 1] = 0;
    }
  }
  (*ic)[m] = cnz_cnt + 1;

  _mm_free(Crow);
  _mm_free(Cflags);
}

#endif

#ifdef UNION_PARALLEL_MERGE
template <typename T>
void my_dcsradd(int m, int n, T *a, int *ja, int *ia, T *b, int *jb, int *ib,
                T **c, int **jc, int **ic, T (*add_fp)(T, T)) {
#ifndef SORTED
#error Merge kernels require sorted inputs
#endif

  int num_threads = omp_get_max_threads();
  assert(num_threads <= omp_get_max_threads());

  (*ic) = reinterpret_cast<int *>(_mm_malloc((m + 1) * sizeof(int), 64));
  int nchunks = num_threads;
  int chunksize = (m + nchunks - 1) / nchunks;
  int *nnzs =
      reinterpret_cast<int *>(_mm_malloc((nchunks + 1) * sizeof(int), 64));
  memset(nnzs, 0, num_threads * sizeof(int));
  T **c_t = new T *[nchunks];
  int **jc_t = new int *[nchunks];

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();

#pragma omp for schedule(dynamic)
    for (int chunk = 0; chunk < nchunks; chunk++) {
      int start_row = chunk * chunksize;
      int end_row = (chunk + 1) * chunksize;
      if (end_row > m) end_row = m;

      // Determine number of nonzeros
      int nnzA = ia[end_row] - ia[start_row];
      int nnzB = ib[end_row] - ib[start_row];
      int nnzmax = nnzA + nnzB;

      // Allocate space for nonzeros
      c_t[chunk] = reinterpret_cast<T *>(
          _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(T), 64));
      jc_t[chunk] = reinterpret_cast<int *>(
          _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(int), 64));

      int cnz_cnt = 0;
      for (int row = start_row; row < end_row; row++) {
        (*ic)[row] = cnz_cnt + 1;

        // Merge c row and tc row into new_c row
        int Astart = ia[row];
        int Aend = ia[row + 1];
        int Bstart = ib[row];
        int Bend = ib[row + 1];

        while ((Astart < Aend) || (Bstart < Bend)) {
          int Acol = (Astart != Aend) ? ja[Astart - 1] : INT_MAX;
          int Bcol = (Bstart != Bend) ? jb[Bstart - 1] : INT_MAX;
          if (Acol < Bcol) {
            c_t[chunk][cnz_cnt] = a[Astart - 1];
            jc_t[chunk][cnz_cnt] = Acol;
            cnz_cnt++;
            Astart++;
          } else if (Bcol < Acol) {
            c_t[chunk][cnz_cnt] = b[Bstart - 1];
            jc_t[chunk][cnz_cnt] = Bcol;
            cnz_cnt++;
            Bstart++;
          } else {
            c_t[chunk][cnz_cnt] = add_fp(a[Astart - 1], b[Bstart - 1]);
            jc_t[chunk][cnz_cnt] = Acol;
            cnz_cnt++;
            Astart++;
            Bstart++;
          }
        }
      }
      nnzs[chunk] = cnz_cnt;
    }  // for each chunk
  }    // pragma omp parallel

  // Main thread allocates a large result array
  int nnzc = 0;
  for (int chunk = 0; chunk < nchunks; chunk++) {
    int tmp = nnzs[chunk];
    nnzs[chunk] = nnzc;
    nnzc += tmp;
  }
  nnzs[nchunks] = nnzc;

  (*c) = reinterpret_cast<T *>(
      _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(T), 64));
  (*jc) = reinterpret_cast<int *>(
      _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(int), 64));

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();

#pragma omp for schedule(dynamic)
    for (int chunk = 0; chunk < nchunks; chunk++) {
      int start_row = chunk * chunksize;
      int end_row = (chunk + 1) * chunksize;
      if (end_row > m) end_row = m;

#pragma simd
      for (int Arow = start_row; Arow < end_row; Arow++) {
        (*ic)[Arow] += nnzs[chunk];
      }
      memcpy((*c) + nnzs[chunk], c_t[chunk],
             (nnzs[chunk + 1] - nnzs[chunk]) * sizeof(T));
      memcpy((*jc) + nnzs[chunk], jc_t[chunk],
             (nnzs[chunk + 1] - nnzs[chunk]) * sizeof(int));
      _mm_free(c_t[chunk]);
      _mm_free(jc_t[chunk]);
    }
  }  // pragma omp parallel

  (*ic)[m] = nnzs[nchunks] + 1;

  delete c_t;
  delete jc_t;
  _mm_free(nnzs);
}

#endif

#ifdef UNION_PARALLEL_SPA

bool cmp_int_union_parallel(int i1, int i2) { return i1 < i2; }

template <typename T>
void my_dcsradd(int m, int n, T *a, int *ja, int *ia, T *b, int *jb, int *ib,
                T **c, int **jc, int **ic, T (*add_fp)(T, T)) {
  int num_threads = omp_get_max_threads();
  assert(num_threads <= omp_get_max_threads());

  (*ic) = reinterpret_cast<int *>(_mm_malloc((m + 1) * sizeof(int), 64));
  int nchunks = num_threads;
  int chunksize = (m + nchunks - 1) / nchunks;
  int *nnzs =
      reinterpret_cast<int *>(_mm_malloc((nchunks + 1) * sizeof(int), 64));
  memset(nnzs, 0, num_threads * sizeof(int));
  T **c_t = new T *[nchunks];
  int **jc_t = new int *[nchunks];

  T **Crow = new T *[num_threads];
  int **Cidxs = new int *[num_threads];
  bool **Cflags = new bool *[num_threads];

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    Crow[tid] = reinterpret_cast<T *>(_mm_malloc(n * sizeof(T), 64));
    Cidxs[tid] = reinterpret_cast<int *>(_mm_malloc(n * sizeof(T), 64));
    Cflags[tid] = reinterpret_cast<bool *>(_mm_malloc(n * sizeof(bool), 64));
    memset(Crow[tid], 0, n * sizeof(T));
    memset(Cidxs[tid], 0, n * sizeof(int));
    memset(Cflags[tid], 0, n * sizeof(bool));

#pragma omp for schedule(dynamic)
    for (int chunk = 0; chunk < nchunks; chunk++) {
      int start_row = chunk * chunksize;
      int end_row = (chunk + 1) * chunksize;
      if (end_row > m) end_row = m;

      // Determine number of nonzeros
      int nnzA = ia[end_row] - ia[start_row];
      int nnzB = ib[end_row] - ib[start_row];
      int nnzmax = nnzA + nnzB;

      // Allocate space for nonzeros
      c_t[chunk] = reinterpret_cast<T *>(
          _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(T), 64));
      jc_t[chunk] = reinterpret_cast<int *>(
          _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(int), 64));

      int cnz_cnt = 0;
      for (int row = start_row; row < end_row; row++) {
        (*ic)[row] = cnz_cnt + 1;
        int c_row_int_start = cnz_cnt;
        int Arow_nnz = 0;
        for (int Anz_id = ia[row]; Anz_id < ia[row + 1]; Anz_id++) {
          int Acol = ja[Anz_id - 1];
          Cidxs[tid][Arow_nnz] = Acol;
          Cflags[tid][Acol - 1] = true;
          Crow[tid][Acol - 1] = a[Anz_id - 1];
          Arow_nnz++;
          cnz_cnt++;
        }
        for (int Bnz_id = ib[row]; Bnz_id < ib[row + 1]; Bnz_id++) {
          int Bcol = jb[Bnz_id - 1];
          if (Cflags[tid][Bcol - 1]) {
            Crow[tid][Bcol - 1] = add_fp(Crow[tid][Bcol - 1], b[Bnz_id - 1]);
          } else {
            Cidxs[tid][Arow_nnz] = Bcol;
            Cflags[tid][Bcol - 1] = true;
            Crow[tid][Bcol - 1] = b[Bnz_id - 1];
            Arow_nnz++;
            cnz_cnt++;
          }
        }
#ifdef SORTED
        std::sort(jc_t[chunk] + c_row_int_start, jc_t[chunk] + cnz_cnt,
                  cmp_int_intersect_parallel);
#endif
        for (int Cnz_id = 0; Cnz_id < Arow_nnz; Cnz_id++) {
          Cflags[tid][Cidxs[tid][Cnz_id] - 1] = 0;
        }

        for (int Cnz_id = c_row_int_start; Cnz_id < cnz_cnt; Cnz_id++) {
          int Ccol = Cidxs[tid][Cnz_id - c_row_int_start];
          jc_t[chunk][Cnz_id] = Ccol;
          c_t[chunk][Cnz_id] = Crow[tid][Ccol - 1];
        }
      }
      nnzs[chunk] = cnz_cnt;
    }  // for each chunk
  }    // pragma omp parallel

  // Main thread allocates a large result array
  int nnzc = 0;
  for (int chunk = 0; chunk < nchunks; chunk++) {
    int tmp = nnzs[chunk];
    nnzs[chunk] = nnzc;
    nnzc += tmp;
  }
  nnzs[nchunks] = nnzc;

  (*c) = reinterpret_cast<T *>(
      _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(T), 64));
  (*jc) = reinterpret_cast<int *>(
      _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(int), 64));

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();

#pragma omp for schedule(dynamic)
    for (int chunk = 0; chunk < nchunks; chunk++) {
      int start_row = chunk * chunksize;
      int end_row = (chunk + 1) * chunksize;
      if (end_row > m) end_row = m;

#pragma simd
      for (int Arow = start_row; Arow < end_row; Arow++) {
        (*ic)[Arow] += nnzs[chunk];
      }
      memcpy((*c) + nnzs[chunk], c_t[chunk],
             (nnzs[chunk + 1] - nnzs[chunk]) * sizeof(T));
      memcpy((*jc) + nnzs[chunk], jc_t[chunk],
             (nnzs[chunk + 1] - nnzs[chunk]) * sizeof(int));
      _mm_free(c_t[chunk]);
      _mm_free(jc_t[chunk]);
    }
  }  // pragma omp parallel
  (*ic)[m] = nnzs[nchunks] + 1;

  delete c_t;
  delete jc_t;
  _mm_free(nnzs);
}

#endif

template <typename T>
void my_denseadd(T *const avalue, bool *const abitvector, T *bvalue,
                 bool *bbitvector, int m, int n, int *nnz, T (*add_fp)(T, T)) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (abitvector[i + j * m]) {
        if (bbitvector[i + j * m]) {
          bvalue[i + j * m] = add_fp(avalue[i + j * m], bvalue[i + j * m]);
        } else {
          bvalue[i + j * m] = avalue[i + j * m];
          nnz++;
        }
        bbitvector[i + j * m] = true;
      }
    }
  }
}


template <typename Ta, typename Tb, typename Tc>
void union_dense_segment(Ta* v1, int * bv1, int nnz, int num_ints, Tb * v2, int * bv2, Tc * v3, int * bv3,
                          void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) 
{

  #pragma omp parallel for
  for (int ii = 0; ii < nnz; ii++) {
    bool set1 = get_bitvector(ii, bv1);
    bool set2 = get_bitvector(ii, bv2);
    if(set1 && !set2)
    {
      v3[ii] = v1[ii];
    }
    else if(!set1 && set2)
    {
      v3[ii] = v2[ii];
    }
    else if(set1 && set2)
    {
      op_fp(v1[ii], v2[ii], &(v3[ii]), vsp);
    }
  }

  #pragma omp parallel for
  for(int i = 0 ; i < num_ints ; i++)
  {
    bv3[i] = bv1[i] | bv2[i];
  }
}

template <typename Ta, typename Tb>
void union_compressed_segment(Ta* v1, int nnz, int capacity, int num_ints, Tb * v2, int * bv2,
                          void (*op_fp)(Ta, Tb, Tb*, void*), void* vsp) 
{
/*
  int * indices = reinterpret_cast<int*>(v1 + nnz);
  for(int i = 0 ; i < nnz ; i++)
  {
    int idx = indices[i];
    if(get_bitvector(idx, bv2))
    {
      Tb tmp = v2[idx];
      op_fp(v1[i], tmp, &(v2[idx]));
    }
    else
    {
      set_bitvector(idx, bv2);
      v2[idx] = v1[i];
    }
  }
  */

  int * indices = reinterpret_cast<int*>(v1 + nnz);
  int npartitions = omp_get_max_threads() * 16;
  #pragma omp parallel for
  for(int p = 0 ; p < npartitions ; p++)
  {
    int nz_per = (nnz + npartitions - 1) / npartitions;
    int start_nnz = p * nz_per;
    int end_nnz = (p+1) * nz_per;
    if(end_nnz > nnz) end_nnz = nnz;

    // Adjust
    if(start_nnz > 0)
    {
      while((start_nnz < nnz) && (indices[start_nnz]/32 == indices[start_nnz-1]/32)) start_nnz++;
    }
    while((end_nnz < nnz) && (indices[end_nnz]/32 == indices[end_nnz-1]/32)) end_nnz++;

    for(int i = start_nnz  ; i < end_nnz ; i++)
    {
      int idx = indices[i];
      if(get_bitvector(idx, bv2))
      {
        Tb tmp = v2[idx];
        op_fp(v1[i], tmp, &(v2[idx]), vsp);
      }
      else
      {
        set_bitvector(idx, bv2);
        v2[idx] = v1[i];
      }
    }
  }
}




#endif  // SRC_SINGLENODE_UNIONREDUCE_H_
