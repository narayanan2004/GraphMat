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


#ifndef SRC_LAYOUTS_H_
#define SRC_LAYOUTS_H_

#include <cmath>
#include <cassert>

void factorize_int(int val, int * res1, int * res2)
{
  int sqrt_val = static_cast<int>((sqrt(static_cast<int>(val)) + 0.5));
  (*res1) = sqrt_val;
  (*res2) = val / sqrt_val;
  while((*res1) * (*res2) != val)
  {
    (*res1)--;
    (*res2) = val / (*res1);
  }
}

int partition_fn_2d(int tileid_x, int tileid_y, int num_tiles_x, int num_tiles_y, int nrank) {
  int nrank_x, nrank_y;
  factorize_int(nrank, &nrank_y, &nrank_x);
  int rank_x = tileid_x % nrank_x;
  int rank_y = tileid_y % nrank_y;
  return rank_y + rank_x * nrank_y;
}

int partition_fn_1d(int tileid_x, int tileid_y, int num_tiles_x, int num_tiles_y, int nrank) {
  return tileid_y % nrank;
}

int vector_partition_fn(int tileid, int ntiles, int nrank) {
  return tileid % nrank;
}

void get_fn_and_tiles(int layout, int nrank, int (**partition_fn)(int,int,int,int,int), int * tiles_per_dim)
{
  assert(layout == 1 || layout == 2 || layout == 3);

  if(layout == 1)
  {
    *tiles_per_dim = nrank;
    *partition_fn = partition_fn_1d;
  }
  if(layout == 2)
  {
    int nrank_x, nrank_y;
    factorize_int(nrank, &nrank_y, &nrank_x);
    *tiles_per_dim = (nrank_x > nrank_y) ? nrank_x : nrank_y;
    *partition_fn = partition_fn_2d;
  }
  if(layout == 3)
  {
    int nrank_x, nrank_y;
    factorize_int(nrank, &nrank_y, &nrank_x);
    *tiles_per_dim = nrank;
    *partition_fn = partition_fn_2d;
  }

}
#endif // SRC_LAYOUTS_H_
