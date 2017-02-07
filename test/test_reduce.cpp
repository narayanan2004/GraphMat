/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ******************************************************************************/
/* Narayanan Sundaram (Intel Corp.), Michael Anderson (Intel Corp.)
 * ******************************************************************************/

#include "catch.hpp"
#include "generator.h"
#include "test_utils.h"

void mapdouble(int * a, int * b, void * vsp) { *b = 2 * (*a); }
void sumreduce(int a, int b, int * c, void * vsp) { *c = a + b; } 

TEST_CASE("reduce", "reduce")
{
  SECTION("mapreduce basic", "mapreduce basic") {
      int tiles_per_dim;
      int (*partition_fn)(int,int,int,int,int);
      GraphMat::get_fn_and_tiles(1, GraphMat::get_global_nrank(), &partition_fn, &tiles_per_dim);
      GraphMat::SpVec<GraphMat::DenseSegment<int> > myvec(1000, tiles_per_dim, GraphMat::vector_partition_fn);
      REQUIRE(myvec.getNNZ() == 0);
      myvec.setAll(1);
      int res = 0;
      GraphMat::MapReduce(&myvec, &res, mapdouble, sumreduce, NULL);
      REQUIRE(res == 2000);
  }
  SECTION("mapreduce empty", "mapreduce empty") {
      int tiles_per_dim;
      int (*partition_fn)(int,int,int,int,int);
      GraphMat::get_fn_and_tiles(1, GraphMat::get_global_nrank(), &partition_fn, &tiles_per_dim);
      GraphMat::SpVec<GraphMat::DenseSegment<int> > myvec(1000, tiles_per_dim, GraphMat::vector_partition_fn);
      REQUIRE(myvec.getNNZ() == 0);
      myvec.set(1, 1);
      myvec.set(10, 1);
      myvec.set(200, 1);
      myvec.set(300, 1);
      int res = 0;
      GraphMat::MapReduce(&myvec, &res, mapdouble, sumreduce, NULL);
      REQUIRE(res == 8);
  }

}

