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


#ifndef SRC_SINGLENODE_FSPGEMM_H_
#define SRC_SINGLENODE_FSPGEMM_H_

#include <algorithm>

#ifdef FSPGEMM_PARALLEL_SPA
bool cmp_int_fspgemm(int i1, int i2) { return i1 < i2; }

template <typename Ta, typename Tb, typename Tc, typename Tf>
void my_fcsrmultcsr(int m, int n, int k, Ta* a, int* ja, int* ia, Tb* b,
                    int* jb, int* ib, Tc** c, int** jc, int** ic, Tc* c_in,
                    int* jc_in, int* ic_in, Tf* f_in, int* jf_in, int* if_in,
                    void (*mul_fp)(Ta, Tb, Tc*, void*), void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp) {
  int num_threads = omp_get_max_threads();
  assert(num_threads <= omp_get_max_threads());

  Tc** Crows = new Tc* [num_threads];
  int** Cidxs = new int* [num_threads];
  bool** Cflags = new bool* [num_threads];
  bool** Fflags = new bool* [num_threads];
  (*ic) = reinterpret_cast<int*>(_mm_malloc((m + 1) * sizeof(int), 64));
  int nchunks = num_threads * 5;
  int chunksize = (m + nchunks - 1) / nchunks;
  int* nnzs =
      reinterpret_cast<int*>(_mm_malloc((nchunks + 1) * sizeof(int), 64));
  memset(nnzs, 0, num_threads * sizeof(int));

  // Flag indicating that we should union the result with another CSR mat
  bool Cin = (c_in != NULL) && (jc_in != NULL) && (ic_in != NULL);

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    Crows[tid] = reinterpret_cast<Tc*>(_mm_malloc(n * sizeof(Tc), 64));
    Cidxs[tid] = reinterpret_cast<int*>(_mm_malloc(n * sizeof(int), 64));
    Cflags[tid] = reinterpret_cast<bool*>(_mm_malloc(n * sizeof(bool), 64));
    memset(Cflags[tid], 0, n * sizeof(bool));
    Fflags[tid] = reinterpret_cast<bool*>(_mm_malloc(n * sizeof(bool), 64));
    memset(Fflags[tid], 0, n * sizeof(bool));

#pragma omp for schedule(dynamic)
    for (int chunk = 0; chunk < nchunks; chunk++) {
      int start_row = chunk * chunksize;
      int end_row = (chunk + 1) * chunksize;
      if (end_row > m) end_row = m;

      // Determine number of nonzeros
      int nnzmax = 0;
      for (int Arow = start_row; Arow < end_row; Arow++) {
        int Arow_nnz = 0;

        // Load values from F into dense row vector
        for (int Fnz_id = if_in[Arow]; Fnz_id < if_in[Arow + 1]; Fnz_id++) {
          int Fcol = jf_in[Fnz_id - 1];
          Fflags[tid][Fcol - 1] = true;
        }

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
            if (Fflags[tid][Bcol - 1] && !Cflags[tid][Bcol - 1]) {
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
        for (int Fnz_id = if_in[Arow]; Fnz_id < if_in[Arow + 1]; Fnz_id++) {
          int Fcol = jf_in[Fnz_id - 1];
          Fflags[tid][Fcol - 1] = false;
        }
      }
      nnzs[chunk] = nnzmax;
    }
    _mm_free(Cidxs[tid]);
#pragma omp barrier
#pragma omp master
    {
      int nnzc = 0;
      for (int chunk = 0; chunk < nchunks; chunk++) {
        int tmp = nnzs[chunk];
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

        // Load values from F into dense row vector
        for (int Fnz_id = if_in[Arow]; Fnz_id < if_in[Arow + 1]; Fnz_id++) {
          int Fcol = jf_in[Fnz_id - 1];
          Fflags[tid][Fcol - 1] = true;
        }

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
            if (Fflags[tid][Bcol - 1] && !Cflags[tid][Bcol - 1]) {
              (*jc)[cnz_cnt] = Bcol;
              mul_fp(a[Anz_id - 1], b[Bnz_id - 1], &(Crows[tid][Bcol-1]), vsp);
              Cflags[tid][Bcol - 1] = true;
              cnz_cnt++;
            } else if (Fflags[tid][Bcol - 1]) {
	      Tc tmp_mul;
              mul_fp(a[Anz_id - 1], b[Bnz_id - 1], &tmp_mul, vsp);
	      Tc tmp_add = Crows[tid][Bcol-1];
              add_fp(
                  tmp_add, tmp_mul, &(Crows[tid][Bcol-1]), vsp);
              Cflags[tid][Bcol - 1] = true;
            }
          }
        }
#ifdef SORTED
        std::sort(*(jc) + c_row_nz_start, (*jc) + cnz_cnt, cmp_int_fspgemm);
#endif
        int num_del = 0;
        for (int Cnz_id = c_row_nz_start; Cnz_id < cnz_cnt; Cnz_id++) {
          num_del++;
          int Ccol = (*jc)[Cnz_id];
          (*c)[Cnz_id] = Crows[tid][Ccol - 1];
          Cflags[tid][Ccol - 1] = false;
        }

        for (int Fnz_id = if_in[Arow]; Fnz_id < if_in[Arow + 1]; Fnz_id++) {
          int Fcol = jf_in[Fnz_id - 1];
          Fflags[tid][Fcol - 1] = false;
        }
      }
    }  // for each chunk

    _mm_free(Crows[tid]);
    _mm_free(Cflags[tid]);
    _mm_free(Fflags[tid]);
  }  // pragma omp parallel

  (*ic)[m] = nnzs[nchunks] + 1;

  delete Crows;
  delete Cflags;
  delete Fflags;
  _mm_free(nnzs);
}
#endif

#endif  // SRC_SINGLENODE_FSPGEMM_H_
