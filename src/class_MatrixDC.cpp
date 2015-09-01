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
/* Narayanan Sundaram (Intel Corp.),  Satya Gautam Vadlamudi (Intel Corp).
 * ******************************************************************************/

// Author: Satya Gautam Vadlamudi
#include "stdio.h"
#include <climits>

template <class E=int>
class MatrixDC {
public:
  // MatrixDC(int _m, int _n, int _nnz, int _isColumn) {
    // m = _m;
    // n = _n;
    // nnz = _nnz;
    // isColumn = _isColumn;
    // nzx = 0;
    // lock = 0;

    // int numx = (m < nnz || nnz == 0) ? m : nnz;  // Assumption for worst-case.

    // long int max = nnz;
    // if (nnz == 0) {
      // printf("Block with zero nnz\n");
      // if (n > 400) {
        // max = m * 400;  // Guess
      // } else {
	// max = m * n;
      // }
    // }
    // //printf("%d %d %ld %d %d %d %ld\n", m, n, nnz, isColumn, nzx, numx, max);
    // xindex = (int*) _mm_malloc(numx*sizeof(int), 64);
    // starty = (int*) _mm_malloc(numx*sizeof(int), 64);
    // yindex = (int*) _mm_malloc(max*sizeof(int), 64);
    // value = (E*) _mm_malloc(max*sizeof(E), 64);
// #ifdef __ASSERT
    // assert(xindex);
    // assert(starty);
    // assert(yindex);
    // assert(value);
// #endif
  // }

  MatrixDC(int _m, int _n, int _nnz, int _isColumn, 
           int _nzx, int * _xindex, int * _starty, 
	   int * _yindex, E * _value) {
    m = _m;
    n = _n;
    nnz = _nnz;
    //isColumn = _isColumn;
    nzx = _nzx;
    //lock = 0;

    xindex = (int*) _xindex;
    starty = (int*) _starty;
    yindex = (int*) _yindex;
    value = (E*) _value;

     // xindexfull = (int*)malloc(sizeof(int)*nnz);
     // for (int i = 0; i < nzx; i++) {
        // int end = (i == nzx - 1 ? nnz : starty[i + 1]);
        // for (int j = starty[i]; j < end; j++) {
          // xindexfull[j] = xindex[i];
        // }
      // }

#ifdef __ASSERT
    assert(xindex);
    assert(starty);
    assert(yindex);
    assert(value);
#endif
  }
 
  
  ~MatrixDC() {
	_mm_free(xindex);
	_mm_free(yindex);
	_mm_free(starty);
	_mm_free(value);
  }

  // static MatrixDC* ReadDCSXRev(int** A, int* Ag, int m, int n, int tid) {
    // int nnz = Ag[tid + 1] - Ag[tid];
    // MatrixDC* M = new MatrixDC(n, m, nnz, false);
    // M->ReadRev(A, Ag[tid]);
    // return M;
  // }

  void print() {
    printf("\n-------------------\n");
    printf("m: %d, n: %d, nzx: %d, nnz: %ld, isColumn: %d\n", m, n, nzx, nnz, isColumn);
    
    for (int i = 0; i < nzx; i++) {
      int end = i == nzx - 1 ? nnz : starty[i + 1];
      for (int j = starty[i]; j < end; j++) {
	printf("%d %d %d\n", xindex[i], yindex[j], value[j]);
      }
    }
    printf("\n-------------------\n");
  }

  void printStats() {
    printf("\n-------------------\n");
    printf("m: %d, n: %d, nzx: %d, nnz: %ld, isColumn: %d\n", m, n, nzx, nnz, isColumn);
    
    int minx, miny, maxx, maxy;
    minx = INT_MAX; miny = INT_MAX;
    maxx = 0; maxy = 0;

    for (int i = 0; i < nzx; i++) {
      int end = i == nzx - 1 ? nnz : starty[i + 1];
      for (int j = starty[i]; j < end; j++) {
	//printf("%d %d %d\n", xindex[i], yindex[j], value[j]);
        minx = std::min(minx, xindex[i]);
        maxx = std::max(maxx, xindex[i]);
        miny = std::min(miny, yindex[j]);
        maxy = std::max(maxy, yindex[j]);
      }
    }
    printf("Range :: X %d to %d, Y %d to %d \n", minx, maxx, miny, maxy);
    printf("\n-------------------\n");
  }

  // void Add(int i, int j, E& v, int* pi, int* xindi, unsigned long* startyi) {
// #ifdef __ASSERT
    // if (i < *pi) {
      // printf("ERROR: i:%d, pi:%d\n", i, *pi);
      // assert(0);
    // }
    // assert(*xindi < m);
    // assert(*startyi < m * 1000);
// #endif
    // if (i == *pi) {
      // yindex[*startyi] = j;
      // value[*startyi] = v;
      // (*startyi)++;
    // } else {
      // *pi = i;
      // (*xindi)++;
      // xindex[*xindi] = i;
      // starty[*xindi] = *startyi;
      // yindex[*startyi] = j;
      // value[*startyi] = v;
      // (*startyi)++;
    // }
    // nzx = *xindi + 1;
    // //printf("nzx in add: %d\n", nzx);
    // nnz++;
  // }

private:
  // void ReadRev(int** A, int offset) {
    // int **M = (int**) _mm_malloc(sizeof(int*) * 3, 64);
    // M[0] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
    // M[1] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
    // M[2] = (int*) _mm_malloc(sizeof(int) * nnz, 64);
    
    // int *t = (int*) _mm_malloc(sizeof(int) * (m + 1), 64);

    // for (int i = 0; i <= m; i++) {
      // t[i] = 0;
    // }
    
    // for (int i = 0; i < nnz; i++) {
      // t[A[1][i + offset]]++;
    // }
    
    // int *j = (int*) _mm_malloc(sizeof(int) * (m + 1), 64);

    // j[0] = 0;
    // for (int i = 1; i <= m; i++) {
      // j[i] = 0;
      // t[i] += t[i-1];
    // }

    // for (int i = 0; i < nnz; i++) {
      // int x = t[A[1][i + offset]-1] + j[A[1][i + offset]];
      // M[0][x] = A[1][i + offset];
      // M[1][x] = A[0][i + offset];
      // M[2][x] = A[2][i + offset];
      // j[A[1][i + offset]]++;
    // }

    // int xindi = -1;
    // unsigned long startyi = 0;
    // int pi = -1;
    // //printf("nnz: %ld\n", nnz);
    // unsigned long _nnz = nnz;
    // nnz = 0;

    // for (int k = 0; k < _nnz; k++) {
      // Add(M[0][k], M[1][k], M[2][k], &pi, &xindi, &startyi);
    // }
// #ifdef __ASSERT
    // assert(_nnz == nnz);
// #endif
    // _mm_free(t);
    // _mm_free(j);
    // _mm_free(M[0]);
    // _mm_free(M[1]);
    // _mm_free(M[2]);
    // _mm_free(M);
  // }

public:
  int m, n, nzx; //isColumn;
  unsigned long nnz;

  int* xindex;
  int* starty;
  int* yindex;
  E* value;
  //int* xindexfull;

  //volatile int lock;
};
