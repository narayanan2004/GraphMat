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


#ifndef SRC_SINGLENODE_INTERSECTREDUCE_H_
#define SRC_SINGLENODE_INTERSECTREDUCE_H_

#include <algorithm>
#include "src/bitvector.h"

#ifdef INTERSECT_MKL
#error
template <typename Ta, typename Tb, typename Tc>
void my_dintersect(int m, int n, Ta *a, int *ja, int *ia, Tb *b, int *jb,
                   int *ib, Tc **c, int **jc, int **ic, Tc (*op_fp)(Ta, Tb)) {
  int nnzc = std::max(ia[m] - 1, ib[m] - 1);
  int nzmax = nnzc;

  (*ic) = reinterpret_cast<int *>(_mm_malloc((m + 1) * sizeof(int), 64));
  (*c) = reinterpret_cast<Tc *>(
      _mm_malloc((uint64_t)nnzc * (uint64_t)sizeof(Tc), 64));
  (*jc) = reinterpret_cast<int *>(
      _mm_malloc((uint64_t)nnzc * (uint64_t)sizeof(int), 64));

  // Add new_c = c + tc
  int cnz_cnt = 0;
  for (int row = 0; row < m; row++) {
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
        Astart++;
      } else if (Bcol < Acol) {
        Bstart++;
      } else {
        (*c)[cnz_cnt] = op_fp(a[Astart - 1], b[Bstart - 1], vsp);
        (*jc)[cnz_cnt] = Acol;
        cnz_cnt++;
        Astart++;
        Bstart++;
      }
    }
  }
  (*ic)[m] = cnz_cnt + 1;
}

#endif

#ifdef INTERSECT_NAIVE_MERGE
template <typename Ta, typename Tb, typename Tc>
void my_dintersect(int m, int n, Ta *a, int *ja, int *ia, Tb *b, int *jb,
                   int *ib, Tc **c, int **jc, int **ic, Tc (*op_fp)(Ta, Tb)) {
#ifndef SORTED
#error Merge kernels require sorted inputs
#endif
  int nnzc = std::max(ia[m] - 1, ib[m] - 1);
  int nzmax = nnzc;

  (*ic) = reinterpret_cast<int *>(_mm_malloc((m + 1) * sizeof(int), 64));
  (*c) = reinterpret_cast<Tc *>(
      _mm_malloc((uint64_t)nnzc * (uint64_t)sizeof(Tc), 64));
  (*jc) = reinterpret_cast<int *>(
      _mm_malloc((uint64_t)nnzc * (uint64_t)sizeof(int), 64));

  // Add new_c = c + tc
  int cnz_cnt = 0;
  for (int row = 0; row < m; row++) {
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
        Astart++;
      } else if (Bcol < Acol) {
        Bstart++;
      } else {
        (*c)[cnz_cnt] = op_fp(a[Astart - 1], b[Bstart - 1], vsp);
        (*jc)[cnz_cnt] = Acol;
        cnz_cnt++;
        Astart++;
        Bstart++;
      }
    }
  }
  (*ic)[m] = cnz_cnt + 1;
}

#endif

#ifdef INTERSECT_PARALLEL_MERGE
template <typename Ta, typename Tb, typename Tc>
void my_dintersect(int m, int n, Ta *a, int *ja, int *ia, Tb *b, int *jb,
                   int *ib, Tc **c, int **jc, int **ic, void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {
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
  Tc **c_t = new Tc *[nchunks];
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
      int nnzmax = std::max(nnzA, nnzB);

      // Allocate space for nonzeros
      c_t[chunk] = reinterpret_cast<Tc *>(
          _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(Tc), 64));
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
            Astart++;
          } else if (Bcol < Acol) {
            Bstart++;
          } else {
            op_fp(a[Astart - 1], b[Bstart - 1], &(c_t[chunk][cnz_cnt]), vsp);
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

  (*c) = reinterpret_cast<Tc *>(
      _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(Tc), 64));
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
             (nnzs[chunk + 1] - nnzs[chunk]) * sizeof(Tc));
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

#ifdef INTERSECT_NAIVE_SPA

bool cmp_int_intersect_naive(int i1, int i2) { return i1 < i2; }

template <typename Ta, typename Tb, typename Tc>
void my_dintersect(int m, int n, Ta *a, int *ja, int *ia, Tb *b, int *jb,
                   int *ib, Tc **c, int **jc, int **ic, Tc (*op_fp)(Ta, Tb)) {
  Tc *Crow = reinterpret_cast<Tc *>(_mm_malloc(n * sizeof(Tc), 64));
  int *Cidxs = reinterpret_cast<int *>(_mm_malloc(n * sizeof(int), 64));
  bool *Cflags = reinterpret_cast<bool *>(_mm_malloc(n * sizeof(bool), 64));
  memset(Crow, 0, n * sizeof(Tc));
  memset(Cflags, 0, n * sizeof(bool));

  int nnzA = ia[m] - 1;
  int nnzB = ib[m] - 1;
  int nnzmax = std::max(nnzA, nnzB);

  (*ic) = reinterpret_cast<int *>(_mm_malloc((m + 1) * sizeof(int), 64));
  (*c) = reinterpret_cast<Tc *>(
      _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(Tc), 64));
  (*jc) = reinterpret_cast<int *>(
      _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(int), 64));

  int cint_cnt = 0;
  for (int Arow = 0; Arow < m; Arow++) {
    int c_row_int_start = cint_cnt;
    int Arow_nnz = 0;
    (*ic)[Arow] = cint_cnt + 1;
    for (int Anz_id = ia[Arow]; Anz_id < ia[Arow + 1]; Anz_id++) {
      int Acol = ja[Anz_id - 1];
      Cidxs[Arow_nnz] = Acol - 1;
      Cflags[Acol - 1] = true;
      Crow[Acol - 1] = a[Anz_id - 1];
      Arow_nnz++;
    }
    for (int Bnz_id = ib[Arow]; Bnz_id < ib[Arow + 1]; Bnz_id++) {
      int Bcol = jb[Bnz_id - 1];
      if (Cflags[Bcol - 1]) {
        (*jc)[cint_cnt] = Bcol;
        cint_cnt++;
        Crow[Bcol - 1] = op_fp(Crow[Bcol - 1], b[Bnz_id - 1], vsp);
      }
    }
#ifdef SORTED
    std::sort((*jc) + c_row_int_start, (*jc) + cint_cnt,
              cmp_int_intersect_naive);
#endif
    for (int Cnz_id = 0; Cnz_id < Arow_nnz; Cnz_id++) {
      Cflags[Cidxs[Cnz_id]] = 0;
    }
    for (int Cnz_id = c_row_int_start; Cnz_id < cint_cnt; Cnz_id++) {
      int Ccol = (*jc)[Cnz_id];
      (*c)[Cnz_id] = Crow[Ccol - 1];
    }
  }
  (*ic)[m] = cint_cnt + 1;

  _mm_free(Cidxs);
  _mm_free(Crow);
  _mm_free(Cflags);
}

#endif

#ifdef INTERSECT_PARALLEL_SPA

bool cmp_int_intersect_parallel(int i1, int i2) { return i1 < i2; }

template <typename Ta, typename Tb, typename Tc>
void my_dintersect(int m, int n, Ta *a, int *ja, int *ia, Tb *b, int *jb,
                   int *ib, Tc **c, int **jc, int **ic, void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {
  int num_threads = omp_get_max_threads();
  assert(num_threads <= omp_get_max_threads());

  (*ic) = reinterpret_cast<int *>(_mm_malloc((m + 1) * sizeof(int), 64));
  int nchunks = num_threads;
  int chunksize = (m + nchunks - 1) / nchunks;
  int *nnzs =
      reinterpret_cast<int *>(_mm_malloc((nchunks + 1) * sizeof(int), 64));
  memset(nnzs, 0, num_threads * sizeof(int));
  Tc **c_t = new Tc *[nchunks];
  int **jc_t = new int *[nchunks];

  Tc **Crow = new Tc *[num_threads];
  int **Cidxs = new int *[num_threads];
  bool **Cflags = new bool *[num_threads];

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    Crow[tid] = reinterpret_cast<Tc *>(_mm_malloc(n * sizeof(Tc), 64));
    Cidxs[tid] = reinterpret_cast<int *>(_mm_malloc(n * sizeof(Tc), 64));
    Cflags[tid] = reinterpret_cast<bool *>(_mm_malloc(n * sizeof(bool), 64));
    memset(Crow[tid], 0, n * sizeof(Tc));
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
      int nnzmax = std::max(nnzA, nnzB);

      // Allocate space for nonzeros
      c_t[chunk] = reinterpret_cast<Tc *>(
          _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(Tc), 64));
      jc_t[chunk] = reinterpret_cast<int *>(
          _mm_malloc((uint64_t)(nnzmax) * (uint64_t)sizeof(int), 64));

      int cint_cnt = 0;
      for (int row = start_row; row < end_row; row++) {
        (*ic)[row] = cint_cnt + 1;
        int c_row_int_start = cint_cnt;
        int Arow_nnz = 0;
        for (int Anz_id = ia[row]; Anz_id < ia[row + 1]; Anz_id++) {
          int Acol = ja[Anz_id - 1];
          Cidxs[tid][Arow_nnz] = Acol - 1;
          Cflags[tid][Acol - 1] = true;
          Crow[tid][Acol - 1] = a[Anz_id - 1];
          Arow_nnz++;
        }
        for (int Bnz_id = ib[row]; Bnz_id < ib[row + 1]; Bnz_id++) {
          int Bcol = jb[Bnz_id - 1];
          if (Cflags[tid][Bcol - 1]) {
            jc_t[chunk][cint_cnt] = Bcol;
            cint_cnt++;
            op_fp(Crow[tid][Bcol - 1], b[Bnz_id - 1], &(Crow[tid][Bcol-1]), vsp);
          }
        }
#ifdef SORTED
        std::sort(jc_t[chunk] + c_row_int_start, jc_t[chunk] + cint_cnt,
                  cmp_int_intersect_parallel);
#endif
        for (int Cnz_id = 0; Cnz_id < Arow_nnz; Cnz_id++) {
          Cflags[tid][Cidxs[tid][Cnz_id]] = 0;
        }
        for (int Cnz_id = c_row_int_start; Cnz_id < cint_cnt; Cnz_id++) {
          int Ccol = jc_t[chunk][Cnz_id];
          c_t[chunk][Cnz_id] = Crow[tid][Ccol - 1];
        }
      }
      nnzs[chunk] = cint_cnt;
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

  (*c) = reinterpret_cast<Tc *>(
      _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(Tc), 64));
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
             (nnzs[chunk + 1] - nnzs[chunk]) * sizeof(Tc));
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

template <typename Ta, typename Tb, typename Tc>
void intersect_dense_segment(Ta* v1, int * bv1, int * nnz, int num_ints, Tb * v2, int * bv2, Tc * v3, int * bv3,
                          void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {

  #pragma omp parallel for
  for(int i = 0 ; i < num_ints ; i++)
  {
    bv3[i] = bv1[i] & bv2[i];
  }

  int tmp_nnz = 0;
  #pragma omp parallel for reduction(+:tmp_nnz)
  for(int ii = 0 ; ii < num_ints ; ii++)
  {
    int cnt = _popcnt32(bv3[ii]);
    if(cnt == 0) continue;
    tmp_nnz += cnt;
    for(int i = ii*32 ; i < (ii+1)*32 ; i++)
    {
      if(get_bitvector(i, bv3))
      {
        Ta tmp = v1[i];
	op_fp(v1[i], v2[i], &(v3[i]), vsp);
      }
    }
  }
  *nnz = tmp_nnz;
}

#endif  // SRC_SINGLENODE_INTERSECTREDUCE_H_
