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

#include <iostream>
#include "catch.hpp"
#include <algorithm>


template<typename T>
bool edge_compare(const GMDP::edge_t<T> &e1,
                  const GMDP::edge_t<T> &e2)
{
        if( (e1.src < e2.src) ||
            ((e1.src == e2.src) && (e1.dst < e2.dst)) ||
            ((e1.src == e2.src) && (e1.dst == e2.dst) && (e1.val < e2.val)) )
        {
                return true;
        }
        return false;
}

template <typename EDGE_T>
void collect_edges(const GMDP::edgelist_t<EDGE_T>& in_edges, GMDP::edgelist_t<EDGE_T>& out_edges) {

    REQUIRE(sizeof(EDGE_T)%sizeof(int) == 0);
    int T_by_int = sizeof(in_edges.edges[0])/sizeof(int);

    int* OERecvCount = new int[GMDP::get_global_nrank()];
    MPI_Allgather(&in_edges.nnz, 1, MPI_INT, OERecvCount, 1, MPI_INT, MPI_COMM_WORLD);

    int* OERecvOffset = new int[GMDP::get_global_nrank()];
    int* OERecvCountInt = new int[GMDP::get_global_nrank()];
    OERecvOffset[0] = 0;
    for (int i = 1; i < GMDP::get_global_nrank(); i++) {
      OERecvOffset[i] = OERecvOffset[i-1] + T_by_int*OERecvCount[i-1];      
    }
    for (int i = 0; i < GMDP::get_global_nrank(); i++) {
      OERecvCountInt[i] = T_by_int*OERecvCount[i];
    }

    int nnz = 0;
    for (int i = 0; i < GMDP::get_global_nrank(); i++) {
      nnz += OERecvCount[i];
    }
    out_edges = GMDP::edgelist_t<EDGE_T>(in_edges.m, in_edges.n, nnz);

    MPI_Allgatherv(in_edges.edges, in_edges.nnz*T_by_int, MPI_INT, out_edges.edges, OERecvCountInt, OERecvOffset, MPI_INT, MPI_COMM_WORLD);

    delete [] OERecvCount;
    delete [] OERecvCountInt;
    delete [] OERecvOffset;
}

template <typename T>
void mul(T a, T b, T * c, void* vsp) {*c = a*b;}

template <typename T>
void add(T a, T b, T * c, void* vsp) {*c = a+b;}

template <typename T>
void max(T a, T b, T * c, void* vsp) {*c = std::max(a,b);}

template <typename T>
void min(T a, T b, T * c, void* vsp) {*c = std::min(a,b);}

