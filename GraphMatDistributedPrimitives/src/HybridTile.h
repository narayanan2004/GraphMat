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


#ifndef SRC_HYBRIDTILE_H_
#define SRC_HYBRIDTILE_H_

#include <string>
#include <algorithm>
#include <vector>
#include "src/CSRTile.h"
#include "src/DCSRTile.h"
#include "src/COOSIMD32Tile.h"

#include "binary_search.h"

template <typename T>
class HybridTile {
 public:
  std::string name;
  int m;
  int n;
  int nnz;
  int nnz1;
  int nnz2;
  DCSRTile<T> * t1;
  COOSIMD32Tile<T> * t2;

  HybridTile() : name("TEMP"), m(0), n(0), nnz(0) {}

  HybridTile(int _m, int _n) : name("TEMP"), m(_m), n(_n), nnz(0) {}

  HybridTile(edge_t<T>* edges, int _m, int _n, int _nnz, int row_start,
          int col_start)
      : name("TEMP"), m(_m), n(_n), nnz(_nnz) {

    __gnu_parallel::sort(edges, edges + nnz, [](const edge_t<T>& a, const edge_t<T>& b) 
    {
      if (a.src < b.src) return true; else if (a.src > b.src) return false;
      if (a.dst < b.dst) return true; else if (a.dst > b.dst) return false;
      return false;
    });

    CSRTile<T> * tmptile = new CSRTile<T>(edges, _m, _n, _nnz, row_start, col_start);
    /*
    unsigned int * ia = new unsigned int[m+1];
    ia[0] = 0;
    int num_partitions = omp_get_max_threads() * 4;
    for(unsigned int i = 0 ; i < nnz-1 ; i++)
    {
      if(edges[i+1].src > edges[i].src)
      {
        unsigned int src_end = (edges[i+1].src - row_start);
        ia[src_end-1] = i+1;
        unsigned int src_start = (edges[i].src - row_start);
        for(unsigned int src = src_start+1 ; src < src_end ; src++)
        {
          ia[src-1] = ia[src_start-1];
        }
      }
    }
    unsigned int last_src = edges[nnz-1].src - row_start;
    for(unsigned int src = last_src ; src <= m ; src++)
    {
      ia[src] = nnz;
    }
    */

    int * ia = tmptile->ia;

    unsigned int * ia_gte16 = new unsigned int[m+1];
    unsigned int * ia_lt16 = new unsigned int[m+1];
    ia_gte16[0] = 0;
    ia_lt16[0] = 0;

    for(unsigned int i = 0 ; i < m ; i++)
    {
      unsigned int nnz_row = ia[i+1]-ia[i];
      if(nnz_row < 16) {
        ia_lt16[i+1] = ia_lt16[i] + nnz_row;
        ia_gte16[i+1] = ia_gte16[i];
      }
      else {
        ia_gte16[i+1] = ia_gte16[i] + nnz_row;
        ia_lt16[i+1] = ia_lt16[i];
      }
    }
    assert(ia_gte16[m] + ia_lt16[m] == nnz);
    nnz1 = ia_gte16[m];
    nnz2 = ia_lt16[m];

    edge_t<T> * edges_gte16 = (edge_t<T>*)_mm_malloc(nnz1 * sizeof(edge_t<T>), 64);
    edge_t<T> * edges_lt16 = (edge_t<T>*)_mm_malloc(nnz2 * sizeof(edge_t<T>), 64);

    for(unsigned int i = 0 ; i < m ; i++)
    {
      unsigned int nnz_row = ia[i+1]-ia[i];
      if(nnz_row < 16) {
        assert(ia_lt16[i] + nnz_row <= nnz2);
        memcpy(edges_lt16 + ia_lt16[i], edges + ia[i] - 1, nnz_row * sizeof(edge_t<T>));
      }
      else {
        assert(ia_gte16[i] + nnz_row <= nnz1);
        memcpy(edges_gte16 + ia_gte16[i], edges + ia[i] - 1, nnz_row * sizeof(edge_t<T>));
      }
    }

    tmptile->clear();
    delete tmptile;

    t1 = new DCSRTile<T>(edges_gte16, _m, _n, nnz1, row_start, col_start);
    t2 = new COOSIMD32Tile<T>(edges_lt16, _m, _n, nnz2, row_start, col_start);

    _mm_free(edges_gte16);
    _mm_free(edges_lt16);
    delete [] ia_gte16;
    delete [] ia_lt16;
  }

  bool isEmpty() const { return nnz <= 0; }

  void get_edges(edge_t<T>* edges, int row_start, int col_start) {
    t1->get_edges(edges, row_start, col_start); 
    t2->get_edges(edges + nnz1, row_start, col_start); 
  }

  HybridTile& operator=(HybridTile other) {
    this->name = other.name;
    this->m = other.m;
    this->n = other.n;
    this->nnz = other.nnz;
    this->nnz1 = other.nnz1;
    this->nnz2 = other.nnz2;
    this->t1 = other.t1;
    this->t2 = other.t2;
  }

  void clear() {
    if (!isEmpty()) {
      t1->clear();
      t2->clear();
    }
    nnz = 0;
  }

  ~HybridTile(void) {}
};

#endif  // SRC_HYBRIDTILE_H_
