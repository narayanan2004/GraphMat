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


#ifndef SRC_GMDP_H_
#define SRC_GMDP_H_

#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <parallel/algorithm>
#include <immintrin.h> // for _mm_malloc
#include <map>
#include <set>
#include <string>
#include <list>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <climits>
#include <sstream>
#include <fstream>
#include <iostream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>


namespace GraphMat {

inline int get_global_nrank() {
  int global_nrank;
  MPI_Comm_size(MPI_COMM_WORLD, &global_nrank);
  return global_nrank;
}

inline int get_global_myrank() {
  int global_myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_myrank);
  return global_myrank;
}

inline double get_compression_threshold() {
  return 0.5;
}

class Serializable {};

#include "GMDP/utils/edgelist.h"
#include "GMDP/matrices/SpMat.h"
#include "GMDP/matrices/layouts.h"
#include "GMDP/matrices/COOSIMD32Tile.h"
#include "GMDP/matrices/COOTile.h"
#include "GMDP/matrices/CSRTile.h"
#include "GMDP/matrices/DCSCTile.h"
#include "GMDP/matrices/HybridTile.h"
#include "GMDP/matrices/SpMat.h"
#include "GMDP/vectors/SpVec.h"
#include "GMDP/vectors/DenseSegment.h"
#include "GMDP/multinode/intersectreduce.h"
#include "GMDP/multinode/reduce.h"
#include "GMDP/multinode/spmspv.h"
#include "GMDP/multinode/spmspv3.h"
#include "GMDP/multinode/applyedges.h"
#include "GMDP/multinode/apply.h"
#include "GMDP/multinode/clear.h"
#include "GMDP/utils/edgelist_transformation.h"


}  // namespace GMDP

#endif  // SRC_GMDP_H_
