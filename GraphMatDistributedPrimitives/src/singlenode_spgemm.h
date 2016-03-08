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


#ifndef SRC_SINGLENODE_SPGEMM_H_
#define SRC_SINGLENODE_SPGEMM_H_

#include <algorithm>
#include "src/bitvector.h"

#ifdef SPGEMM_NAIVE_SPA
bool cmp_int_spgemm_naive(int i1, int i2) { return i1 < i2; }
template <typename Ta, typename Tb, typename Tc>
void my_dcsrmultcsr(int m, int n, int k, Ta* a, int* ja, int* ia, Tb* b,
                    int* jb, int* ib, Tc** c, int** jc, int** ic, Tc* c_in,
                    int* jc_in, int* ic_in, void (*mul_fp)(Ta, Tb, Tc*, void*),
                    void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
  Tc* Crow = reinterpret_cast<Tc*>(_mm_malloc(n * sizeof(Tc), 64));
  int* Cidxs = reinterpret_cast<int*>(_mm_malloc(n * sizeof(int), 64));
  bool* Cflags = reinterpret_cast<bool*>(_mm_malloc(n * sizeof(bool), 64));
  memset(Crow, 0, n * sizeof(Tc));
  memset(Cflags, 0, n * sizeof(bool));

  // Flag indicating that we should union the result with another CSR mat
  bool Cin = (c_in != NULL) && (jc_in != NULL) && (ic_in != NULL);

  int nnzc = 0;
  for (int Arow = 0; Arow < m; Arow++) {
    int Arow_nnz = 0;

    // Load values from C_in into dense row vector
    if (Cin) {
      int row_nnz = 0;
      for (int Cnz_id = ic_in[Arow]; Cnz_id < ic_in[Arow + 1]; Cnz_id++) {
        int Ccol = jc_in[Cnz_id - 1];
        Cidxs[Arow_nnz] = Ccol - 1;
        Cflags[Ccol - 1] = true;
        row_nnz++;
        Arow_nnz++;
      }
      nnzc += row_nnz;
    }

    for (int Anz_id = ia[Arow]; Anz_id < ia[Arow + 1]; Anz_id++) {
      int Acol = ja[Anz_id - 1];
      int row_nnz = 0;
      for (int Bnz_id = ib[Acol - 1]; Bnz_id < ib[Acol]; Bnz_id++) {
        int Bcol = jb[Bnz_id - 1];
        if (!Cflags[Bcol - 1]) {
          Cidxs[Arow_nnz] = Bcol - 1;
          Cflags[Bcol - 1] = true;
          row_nnz++;
          Arow_nnz++;
        }
      }
      nnzc += row_nnz;
    }
    for (int idx = 0; idx < Arow_nnz; idx++) {
      Cflags[Cidxs[idx]] = false;
    }
  }

  _mm_free(Cidxs);

  (*ic) = reinterpret_cast<int*>(_mm_malloc((m + 1) * sizeof(int), 64));
  (*c) = reinterpret_cast<Tc*>(
      _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(Tc), 64));
  (*jc) = reinterpret_cast<int*>(
      _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(int), 64));

  // Multiply tc = ta * tb
  int cnz_cnt = 0;
  for (int Arow = 0; Arow < m; Arow++) {
    int c_row_nz_start = cnz_cnt;
    (*ic)[Arow] = cnz_cnt + 1;

    // Load values from C_in into dense row vector
    if (Cin) {
      for (int Cnz_id = ic_in[Arow]; Cnz_id < ic_in[Arow + 1]; Cnz_id++) {
        int Ccol = jc_in[Cnz_id - 1];
        (*jc)[cnz_cnt] = Ccol;
        cnz_cnt++;
        Cflags[Ccol - 1] = 1;
        Crow[Ccol - 1] = c_in[Cnz_id - 1];
      }
    }

    for (int Anz_id = ia[Arow]; Anz_id < ia[Arow + 1]; Anz_id++) {
      int Acol = ja[Anz_id - 1];
      for (int Bnz_id = ib[Acol - 1]; Bnz_id < ib[Acol]; Bnz_id++) {
        int Bcol = jb[Bnz_id - 1];
        // if(Crow[Bcol-1] == 0.0)
        if (Cflags[Bcol - 1] == 0) {
          (*jc)[cnz_cnt] = Bcol;
          cnz_cnt++;
        }
        Cflags[Bcol - 1] = 1;
        // Crow[Bcol-1] += a[Anz_id-1] * b[Bnz_id-1];
	Tc mul_tmp;
        mul_fp(a[Anz_id - 1], b[Bnz_id - 1], &mul_tmp, vsp)
	Tc add_tmp = Crow[Bcol-1];
        add_fp(add_tmp, mul_tmp, &(Crow[Bcol-1]), vsp);
#ifdef COUNT_FLOPS
        mul_flops++;
        add_flops++;
#endif
      }
    }
#ifdef SORTED
    std::sort((*jc) + c_row_nz_start, (*jc) + cnz_cnt, cmp_int_spgemm_naive);
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

#ifdef SPGEMM_PARALLEL_SPA
bool cmp_int(int i1, int i2) { return i1 < i2; }

template <typename Ta, typename Tb, typename Tc>
void my_dcsrmultcsr(int m, int n, int k, Ta* a, int* ja, int* ia, Tb* b,
                    int* jb, int* ib, Tc** c, int** jc, int** ic, Tc* c_in,
                    int* jc_in, int* ic_in, void (*mul_fp)(Ta, Tb, Tc*, void*),
                    void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
  int num_threads = omp_get_max_threads();
  assert(num_threads <= omp_get_max_threads());

  Tc** Crows = new Tc* [num_threads];
  int** Cidxs = new int* [num_threads];
  bool** Cflags = new bool* [num_threads];
  (*ic) = reinterpret_cast<int*>(_mm_malloc((m + 1) * sizeof(int), 64));
  int nchunks = num_threads * 5;
  int chunksize = (m + nchunks - 1) / nchunks;
  uint64_t * nnzs =
      reinterpret_cast<uint64_t*>(_mm_malloc((nchunks + 1) * sizeof(uint64_t), 64));
  memset(nnzs, 0, num_threads * sizeof(uint64_t));

  // Flag indicating that we should union the result with another CSR mat
  bool Cin = (c_in != NULL) && (jc_in != NULL) && (ic_in != NULL);

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    Crows[tid] = reinterpret_cast<Tc*>(_mm_malloc(n * sizeof(Tc), 64));
    Cidxs[tid] = reinterpret_cast<int*>(_mm_malloc(n * sizeof(int), 64));
    Cflags[tid] = reinterpret_cast<bool*>(_mm_malloc(n * sizeof(bool), 64));
    memset(Cflags[tid], 0, n * sizeof(bool));

#pragma omp for schedule(dynamic)
    for (int chunk = 0; chunk < nchunks; chunk++) {
      int start_row = chunk * chunksize;
      int end_row = (chunk + 1) * chunksize;
      if (end_row > m) end_row = m;

      // Determine number of nonzeros
      uint64_t nnzmax = 0;
      for (int Arow = start_row; Arow < end_row; Arow++) {
        int Arow_nnz = 0;

        // Load values from C_in into dense row vector
        if (Cin) {
          int row_nnz = 0;
          for (int Cnz_id = ic_in[Arow]; Cnz_id < ic_in[Arow + 1]; Cnz_id++) {
            int Ccol = jc_in[Cnz_id - 1];
            Cidxs[tid][Arow_nnz] = Ccol - 1;
            Cflags[tid][Ccol - 1] = true;
            row_nnz++;
            Arow_nnz++;
          }
          nnzmax += row_nnz;
        }

        for (int Anz_id = ia[Arow]; Anz_id < ia[Arow + 1]; Anz_id++) {
          int Acol = ja[Anz_id - 1];
          int row_nnz = 0;
          for (int Bnz_id = ib[Acol - 1]; Bnz_id < ib[Acol]; Bnz_id++) {
            int Bcol = jb[Bnz_id - 1];
            if (!Cflags[tid][Bcol - 1]) {
              Cidxs[tid][Arow_nnz] = Bcol - 1;
              Cflags[tid][Bcol - 1] = true;
              row_nnz++;
              Arow_nnz++;
            }
          }
          nnzmax += row_nnz;
        }
        for (int idx = 0; idx < Arow_nnz; idx++) {
          Cflags[tid][Cidxs[tid][idx]] = false;
        }
      }
      nnzs[chunk] = nnzmax;
    }
    _mm_free(Cidxs[tid]);
#pragma omp barrier
#pragma omp master
    {
      uint64_t nnzc = 0;
      for (int chunk = 0; chunk < nchunks; chunk++) {
        uint64_t tmp = nnzs[chunk];
        nnzs[chunk] = nnzc;
        nnzc += tmp;
      }
      nnzs[nchunks] = nnzc;
      (*c) = reinterpret_cast<Tc*>(
          _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(Tc), 64));
      (*jc) = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(int), 64));
    }
#pragma omp barrier
#pragma omp for schedule(dynamic)
    for (int chunk = 0; chunk < nchunks; chunk++) {
      int start_row = chunk * chunksize;
      int end_row = (chunk + 1) * chunksize;
      if (end_row > m) end_row = m;

      // Perform multiplication
      int cnz_cnt = nnzs[chunk];
      for (int Arow = start_row; Arow < end_row; Arow++) {
        int c_row_nz_start = cnz_cnt;
        (*ic)[Arow] = cnz_cnt + 1;

        // Load values from C_in into dense row vector
        if (Cin) {
          for (int Cnz_id = ic_in[Arow]; Cnz_id < ic_in[Arow + 1]; Cnz_id++) {
            int Ccol = jc_in[Cnz_id - 1];
            (*jc)[cnz_cnt] = Ccol;
            cnz_cnt++;
            Cflags[tid][Ccol - 1] = 1;
            Crows[tid][Ccol - 1] = c_in[Cnz_id - 1];
          }
        }

        for (int Anz_id = ia[Arow]; Anz_id < ia[Arow + 1]; Anz_id++) {
          int Acol = ja[Anz_id - 1];
          for (int Bnz_id = ib[Acol - 1]; Bnz_id < ib[Acol]; Bnz_id++) {
            int Bcol = jb[Bnz_id - 1];
            if (!Cflags[tid][Bcol - 1]) {
              (*jc)[cnz_cnt] = Bcol;
              mul_fp(a[Anz_id - 1], b[Bnz_id - 1], &(Crows[tid][Bcol-1]), vsp);
              cnz_cnt++;
            } else {
	      Tc tmp_mul;
              mul_fp(a[Anz_id - 1], b[Bnz_id - 1], &tmp_mul, vsp);
	      Tc tmp_add = Crows[tid][Bcol-1];
              add_fp(
                  tmp_add, tmp_mul, &(Crows[tid][Bcol-1]), vsp);
            }
            Cflags[tid][Bcol - 1] = true;
          }
        }
#ifdef SORTED
        std::sort((*jc) + c_row_nz_start, (*jc) + cnz_cnt, cmp_int);
#endif
        int num_del = 0;
        for (int Cnz_id = c_row_nz_start; Cnz_id < cnz_cnt; Cnz_id++) {
          num_del++;
          int Ccol = (*jc)[Cnz_id];
          (*c)[Cnz_id] = Crows[tid][Ccol - 1];
          Cflags[tid][Ccol - 1] = false;
        }
      }
    }  // for each chunk

    _mm_free(Crows[tid]);
    _mm_free(Cflags[tid]);
  }  // pragma omp parallel

  (*ic)[m] = nnzs[nchunks] + 1;

  delete Crows;
  delete Cflags;
  _mm_free(nnzs);
}
#endif

template <typename Ta, typename Tb, typename Tc>
void my_dcscmultdense(int* row_inds, int* col_ptrs, int* col_indices, Ta* vals,
                      int num_partitions, int* row_pointers, int* col_starts,
                      int* edge_pointers, Tb* bvalue, int * bbitvector,
                      Tc* cvalue, int * cbitvector, int m, int n, int k,
                      int* nnz, void (*op_mul)(Ta, Tb, Tc*, void*), void (*op_add)(Tc, Tc, Tc*, void*), void* vsp) {
  int* new_nnzs = new int[num_partitions];
  memset(new_nnzs, 0, num_partitions * sizeof(int));

#pragma omp parallel for
  for (int p = 0; p < num_partitions; p++) {
    const int* column_offset = col_indices + col_starts[p];
    const int* partitioned_row_offset = row_inds + edge_pointers[p];
    const Ta* partitioned_val_offset = vals + edge_pointers[p];
    const int* col_ptrs_cur = col_ptrs + col_starts[p];

    // For each column
    for (int j = 0; j < (col_starts[p + 1] - col_starts[p]) - 1; j++) {
      int col_index = col_indices[col_starts[p] + j];

      // For each B column
//          _mm_prefetch((char*)(bvalue + column_offset[j + 4] + jj * k), _MM_HINT_T0);
      int nz_idx = col_ptrs_cur[j];
      for ( ; nz_idx < col_ptrs_cur[j+1] ; nz_idx++) {
        int row_ind = partitioned_row_offset[nz_idx];
        Ta Aval = partitioned_val_offset[nz_idx];
        for (int jj = 0; jj < n; jj++) {
          if (get_bitvector(col_index + jj * k, bbitvector)) {
            Tb Bval = bvalue[col_index + jj * k];
            if (get_bitvector(row_ind + jj * m, cbitvector)) {
	      Tc mul_tmp;
	      op_mul(Aval, Bval, &mul_tmp);
	      Tc add_tmp = cvalue[row_ind + jj * m];
	      op_add(add_tmp, mul_tmp, &(cvalue[row_ind + jj * m]));
            } else {
              op_mul(Aval, Bval, &(cvalue[row_ind + jj * m]));
              new_nnzs[p]++;
            }
	    set_bitvector(row_ind + jj * m, cbitvector);
          }
        }
      }
    }
  }
  for (int p = 0; p < num_partitions; p++) {
    *nnz += new_nnzs[p];
  }
}


#ifdef SPGEMM_PARALLEL_MERGE
template <typename Tc>
void merge(Tc* a, int* ja, int Aend, Tc* b, int* jb, int Bend, Tc* c, int* jc,
           int* Cend, void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
  // Merge c row and tc row into new_c row
  int Astart = 0;
  int Bstart = 0;
  int cnz_cnt = 0;

  while ((Astart < Aend) || (Bstart < Bend)) {
    int Acol = (Astart != Aend) ? ja[Astart] : INT_MAX;
    int Bcol = (Bstart != Bend) ? jb[Bstart] : INT_MAX;
    if (Acol < Bcol) {
      c[cnz_cnt] = a[Astart];
      jc[cnz_cnt] = Acol;
      cnz_cnt++;
      Astart++;
    } else if (Bcol < Acol) {
      c[cnz_cnt] = b[Bstart];
      jc[cnz_cnt] = Bcol;
      cnz_cnt++;
      Bstart++;
    } else {
      add_fp(a[Astart], b[Bstart], &(c[cnz_cnt]), vsp);
      jc[cnz_cnt] = Acol;
      cnz_cnt++;
      Astart++;
      Bstart++;
    }
  }
  *Cend = cnz_cnt;
}

template <typename Tc>
void merge_sort(Tc* c_buf[2], int* jc_buf[2], int* current_buf, int* row_ptrs,
                int row_cnt, void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
  int cur_row_cnt = row_cnt;
  int result_ptr = 0;
  while (cur_row_cnt > 1) {
    // For each pair
    result_ptr = 0;
    int new_row_cnt = 0;
    for (int r = 0; r < cur_row_cnt; r += 2) {
      if (cur_row_cnt - r > 1) {
        int Clen = 0;
        merge<Tc>(c_buf[(*current_buf)] + row_ptrs[r],
                  jc_buf[(*current_buf)] + row_ptrs[r],
                  row_ptrs[r + 1] - row_ptrs[r],
                  c_buf[(*current_buf)] + row_ptrs[r + 1],
                  jc_buf[(*current_buf)] + row_ptrs[r + 1],
                  row_ptrs[r + 2] - row_ptrs[r + 1],
                  c_buf[1 - (*current_buf)] + result_ptr,
                  jc_buf[1 - (*current_buf)] + result_ptr, &Clen, add_fp, vsp);
        row_ptrs[r / 2] = result_ptr;
        row_ptrs[r / 2 + 1] = result_ptr + Clen;
        result_ptr += Clen;
      } else {
        int Clen = (row_ptrs[r + 1] - row_ptrs[r]);
        memcpy(c_buf[1 - (*current_buf)] + result_ptr,
               c_buf[(*current_buf)] + row_ptrs[r], Clen * sizeof(Tc));
        memcpy(jc_buf[1 - (*current_buf)] + result_ptr,
               jc_buf[(*current_buf)] + row_ptrs[r], Clen * sizeof(int));
        row_ptrs[r / 2] = result_ptr;
        row_ptrs[r / 2 + 1] = result_ptr + Clen;
        result_ptr += Clen;
      }
      new_row_cnt++;
    }
    cur_row_cnt = new_row_cnt;
    (*current_buf) = 1 - (*current_buf);
  }
}

template <typename Ta, typename Tb, typename Tc>
void my_dcsrmultcsr(int m, int n, int k, Ta* a, int* ja, int* ia, Tb* b,
                    int* jb, int* ib, Tc** c, int** jc, int** ic, Tc* c_in,
                    int* jc_in, int* ic_in, void (*mul_fp)(Ta, Tb, Tc*, void*),
                    void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
#ifndef SORTED
#error Merge kernels require sorted inputs
#endif

  int num_threads = omp_get_max_threads();
  assert(num_threads <= omp_get_max_threads());

  int** Cidxs = new int* [num_threads];
  bool** Cflags = new bool* [num_threads];

  (*ic) = reinterpret_cast<int*>(_mm_malloc((m + 1) * sizeof(int), 64));
  int nchunks = num_threads * 5;
  int chunksize = (m + nchunks - 1) / nchunks;
  int row_buf_len;
  int row_ptr_len;
  int* nnzs =
      reinterpret_cast<int*>(_mm_malloc((nchunks + 1) * sizeof(int), 64));
  int* max_row_ubounds =
      reinterpret_cast<int*>(_mm_malloc((nchunks + 1) * sizeof(int), 64));
  int* max_row_nums =
      reinterpret_cast<int*>(_mm_malloc((nchunks + 1) * sizeof(int), 64));
  memset(nnzs, 0, num_threads * sizeof(int));
  memset(max_row_ubounds, 0, num_threads * sizeof(int));
  memset(max_row_nums, 0, num_threads * sizeof(int));

  // Flag indicating that we should union the result with another CSR mat
  bool Cin = (c_in != NULL) && (jc_in != NULL) && (ic_in != NULL);

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    Cidxs[tid] = reinterpret_cast<int*>(_mm_malloc(n * sizeof(int), 64));
    Cflags[tid] = reinterpret_cast<bool*>(_mm_malloc(n * sizeof(bool), 64));
    memset(Cflags[tid], 0, n * sizeof(bool));

#pragma omp for schedule(dynamic)
    for (int chunk = 0; chunk < nchunks; chunk++) {
      int start_row = chunk * chunksize;
      int end_row = (chunk + 1) * chunksize;
      if (end_row > m) end_row = m;

      // Determine number of nonzeros
      int nnzmax = 0;
      int max_row_ub = 0;
      int max_num_rows = 0;
      for (int Arow = start_row; Arow < end_row; Arow++) {
        int row_ub = 0;
        int Arow_nnz = 0;
        max_num_rows = std::max(max_num_rows, ia[Arow + 1] - ia[Arow]);

        // Load values from C_in into dense row vector
        if (Cin) {
          row_ub += (ic_in[Arow + 1] - ic_in[Arow]);
          int row_nnz = 0;
          for (int Cnz_id = ic_in[Arow]; Cnz_id < ic_in[Arow + 1]; Cnz_id++) {
            int Ccol = jc_in[Cnz_id - 1];
            Cidxs[tid][Arow_nnz] = Ccol - 1;
            Cflags[tid][Ccol - 1] = true;
            row_nnz++;
            Arow_nnz++;
          }
          nnzmax += row_nnz;
        }

        for (int Anz_id = ia[Arow]; Anz_id < ia[Arow + 1]; Anz_id++) {
          int Acol = ja[Anz_id - 1];
          row_ub += (ib[Acol] - ib[Acol - 1]);

          int row_nnz = 0;
          for (int Bnz_id = ib[Acol - 1]; Bnz_id < ib[Acol]; Bnz_id++) {
            int Bcol = jb[Bnz_id - 1];
            if (!Cflags[tid][Bcol - 1]) {
              Cidxs[tid][Arow_nnz] = Bcol - 1;
              Cflags[tid][Bcol - 1] = true;
              row_nnz++;
              Arow_nnz++;
            }
          }
          nnzmax += row_nnz;
        }
        (*ic)[Arow] = Arow_nnz;
        max_row_ub = std::max(max_row_ub, row_ub);
        for (int idx = 0; idx < Arow_nnz; idx++) {
          Cflags[tid][Cidxs[tid][idx]] = false;
        }
      }

      nnzs[chunk] = nnzmax;
      max_row_ubounds[chunk] = max_row_ub;
      max_row_nums[chunk] = max_num_rows;
    }
    _mm_free(Cidxs[tid]);
#pragma omp barrier
#pragma omp master
    {
      int nnzc = 0;
      row_buf_len = 0;
      for (int chunk = 0; chunk < nchunks; chunk++) {
        int tmp = nnzs[chunk];
        nnzs[chunk] = nnzc;
        nnzc += tmp;
        row_buf_len = std::max(row_buf_len, max_row_ubounds[chunk]);
        row_ptr_len = std::max(row_ptr_len, max_row_nums[chunk]);
      }
      nnzs[nchunks] = nnzc;
      (*c) = reinterpret_cast<Tc*>(
          _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(Tc), 64));
      (*jc) = reinterpret_cast<int*>(
          _mm_malloc((uint64_t)(nnzc) * (uint64_t)sizeof(int), 64));
    }
#pragma omp barrier

    // Allocate row buffers
    Tc* c_buf[2];
    int* jc_buf[2];
    c_buf[0] = reinterpret_cast<Tc*>(_mm_malloc(row_buf_len * sizeof(Tc), 64));
    c_buf[1] = reinterpret_cast<Tc*>(_mm_malloc(row_buf_len * sizeof(Tc), 64));
    jc_buf[0] =
        reinterpret_cast<int*>(_mm_malloc(row_buf_len * sizeof(int), 64));
    jc_buf[1] =
        reinterpret_cast<int*>(_mm_malloc(row_buf_len * sizeof(int), 64));
    int* row_ptrs =
        reinterpret_cast<int*>(_mm_malloc((row_ptr_len + 1) * sizeof(int), 64));

#pragma omp for schedule(dynamic)
    for (int chunk = 0; chunk < nchunks; chunk++) {
      int start_row = chunk * chunksize;
      int end_row = (chunk + 1) * chunksize;
      if (end_row > m) end_row = m;

      // Perform multiplication
      int cnz_cnt = nnzs[chunk];
      for (int Arow = start_row; Arow < end_row; Arow++) {
        int buf_nz_cnt = 0;
        int c_row_nz_start = cnz_cnt;
        int row_cnt = 0;
        int Arow_nnz = (*ic)[Arow];
        (*ic)[Arow] = cnz_cnt + 1;

        if (Cin) {
          row_ptrs[row_cnt] = buf_nz_cnt;
          for (int Cnz_id = ic_in[Arow]; Cnz_id < ic_in[Arow + 1]; Cnz_id++) {
            c_buf[0][buf_nz_cnt] = c_in[Cnz_id - 1];
            jc_buf[0][buf_nz_cnt] = jc_in[Cnz_id - 1];
            buf_nz_cnt++;
          }
          row_cnt++;
          row_ptrs[row_cnt] = buf_nz_cnt;
        }

        for (int Anz_id = ia[Arow]; Anz_id < ia[Arow + 1]; Anz_id++) {
          int Acol = ja[Anz_id - 1];
          Ta Aval = a[Anz_id - 1];
          row_ptrs[row_cnt] = buf_nz_cnt;
          // Copy B row into c_buf[0] and jc_buf[0]
          for (int Bnz_id = ib[Acol - 1]; Bnz_id < ib[Acol]; Bnz_id++) {
            mul_fp(Aval, b[Bnz_id - 1], &(c_buf[0][buf_nz_cnt]), vsp);
            jc_buf[0][buf_nz_cnt] = jb[Bnz_id - 1];
            buf_nz_cnt++;
          }
          row_cnt++;
        }
        row_ptrs[row_cnt] = buf_nz_cnt;
        cnz_cnt += Arow_nnz;

        // Merge sort
        int current_buf = 0;
        merge_sort<Tc>(c_buf, jc_buf, &current_buf, row_ptrs, row_cnt, add_fp);
        memcpy((*c) + c_row_nz_start, c_buf[current_buf],
               Arow_nnz * sizeof(Tc));
        memcpy((*jc) + c_row_nz_start, jc_buf[current_buf],
               Arow_nnz * sizeof(int));
      }
    }  // for each chunk
    _mm_free(c_buf[0]);
    _mm_free(c_buf[1]);
    _mm_free(jc_buf[0]);
    _mm_free(jc_buf[1]);
    _mm_free(row_ptrs);
    _mm_free(Cflags[tid]);
  }  // pragma omp parallel

  (*ic)[m] = nnzs[nchunks] + 1;

  delete[] Cflags;
  _mm_free(nnzs);
  _mm_free(max_row_ubounds);
}

#endif


#endif  // SRC_SINGLENODE_SPGEMM_H_



