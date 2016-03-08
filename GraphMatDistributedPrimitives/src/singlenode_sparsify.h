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


#ifndef SRC_SINGLENODE_SPARSIFY_H_
#define SRC_SINGLENODE_SPARSIFY_H_

template <typename Ta, typename Tb>
void sparsify_dense_segment(Ta* v1, int * bitvector, int nnz, int num_ints,
                         Tb* v2, int * bitvector2, int * nnz_out,
                          void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp) {
  #pragma omp parallel for
  for(int i = 0 ; i < num_ints ; i++)
  {
    bitvector2[i] = 0;
  }

  int sparse_nnz = 0;
  #pragma omp parallel for reduction(+:sparse_nnz)
  for(int ii = 0 ; ii < num_ints ; ii++)
  {
    int local_nnz = 0;
    if(_popcnt32(bitvector[ii] == 0)) continue;
    for(int i = ii*32 ; i < (ii+1)*32 ; i++)
    {
      if(get_bitvector(i, bitvector))
      {
        bool keep = true;
	add_fp(v1[i], &keep, &(v2[i]), vsp);
	if(keep)
	{
	  set_bitvector(i, bitvector2);
	  local_nnz++;
	}
      }
    }
    sparse_nnz += local_nnz;
  }
  *nnz_out = sparse_nnz;
}

template <typename Ta, typename Tb>
void sparsify_dense_tile(Ta* v1, int * bitvector, int m, int n, 
                         Tb* v2, int * bitvector2, int * nnz_out,
                          void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp) {
  int nnz = m*n;
  int sparse_nnz = 0;
  #pragma omp parallel for reduction(+:sparse_nnz)
  for(int i = 0 ; i < nnz ; i+=256)
  {
    int local_nnz = 0;
    int lim = i + 256;
    if(lim > nnz) lim = m;
    for(int ii = i ; ii < lim ; ii++)
    {
      if(get_bitvector(ii, bitvector))
      {
        bool keep = true;
        add_fp(v1[ii], &keep, &(v2[ii]), vsp);
        if(keep)
        {
          set_bitvector(ii, bitvector2);
	  local_nnz++;
        }
      }
    }
    sparse_nnz += local_nnz;
  }
  *nnz_out = sparse_nnz;
}

template <typename Ta, typename Tb>
void sparsify_dense_to_csr(Ta* v1, int * bitvector, int m, int n, 
                           Tb ** ac, int ** jc, int ** ic,
                          void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp) {

  // Allocate B to be at least as large as A

  int nthreads = omp_get_max_threads();
  int npartitions = nthreads * 4;

  int * nnzp = new int[npartitions];
  #pragma omp parallel for schedule(dynamic)
  for(int p = 0 ; p < npartitions ; p++)
  {
    int local_nnz = 0;
    int rows_per_partition = ((m + npartitions-1) / npartitions);
    int start_m = p * rows_per_partition;
    int end_m = (p+1) * rows_per_partition;
    if(end_m > m) end_m = m;

    for(int i = start_m ; i < end_m ; i++)
    {
      for(int j = 0 ; j < n ; j++)
      {
        int idx = i + j * m;
        if(get_bitvector(idx, bitvector))
	{
	  Tb tmp;
	  bool keep = true;
	  add_fp(v1[idx], &keep, &tmp, vsp);
	  if(keep)
	  {
	    local_nnz++;
	  }
	}
      }
    }
    nnzp[p] = local_nnz;
  }

  int nnz = 0;
  for(int p = 0 ; p < npartitions ; p++)
  {
    nnz += nnzp[p];
  }

  for(int p = 1 ; p < npartitions ; p++)
  {
    nnzp[p] = nnzp[p-1] + nnzp[p];
  }
  *ac = reinterpret_cast<Tb*> (_mm_malloc(nnz * sizeof(Tb), 64));
  *jc = reinterpret_cast<int*> (_mm_malloc(nnz* sizeof(int), 64));
  *ic = reinterpret_cast<int*>( _mm_malloc((m+1) * sizeof(int), 64));

  #pragma omp parallel for schedule(dynamic)
  for(int p = 0 ; p < npartitions ; p++)
  {
    int start_nnz = 0;
    if(p > 0) start_nnz = nnzp[p-1];
    int rows_per_partition = ((m + npartitions-1) / npartitions);
    int start_m = p * rows_per_partition;
    int end_m = (p+1) * rows_per_partition;
    if(end_m > m) end_m = m;
    for(int i = start_m ; i < end_m ; i++)
    {
      (*ic)[i] = start_nnz+1;
      for(int j = 0 ; j < n ; j++)
      {
        int idx = i + j * m;
        if(get_bitvector(idx, bitvector))
	{
	  Tb tmp;
	  bool keep = true;
	  add_fp(v1[idx], &keep, &tmp, vsp);
	  if(keep)
	  {
	    (*ac)[start_nnz] = tmp;
	    (*jc)[start_nnz] = j+1;
	    start_nnz++;
	  }
	}
      }
    }
  }

  (*ic)[m] = nnz+1;

  delete [] nnzp;
}






#endif  // SRC_SINGLENODE_SPARSIFY_H_
