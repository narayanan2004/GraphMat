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
#include "generator.h"
#include <algorithm>
#include "test_utils.h"
#include <vector>
#include "boost/serialization/vector.hpp"

class sv2 : public GraphMat::Serializable {
  public:
    std::vector<int> v;
  public:
    sv2() {}
    void clear() {
      v.clear();
    }
    void push_back(int t) {
      v.push_back(t);
    }
    friend boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & v;
    }
};

TEST_CASE("vector", "vector")
{
  SECTION(" SpVec basic tests", "SpVec basic tests") {
      int tiles_per_dim;
      int (*partition_fn)(int,int,int,int,int);
      GraphMat::get_fn_and_tiles(1, GraphMat::get_global_nrank(), &partition_fn, &tiles_per_dim);
      GraphMat::SpVec<GraphMat::DenseSegment<int> > myvec(1000, tiles_per_dim, GraphMat::vector_partition_fn);
      REQUIRE(myvec.getNNZ() == 0);

      myvec.set(1, 1);
      myvec.set(10, 1);
      myvec.set(200, 1);
      myvec.set(300, 1);

      REQUIRE(myvec.getNNZ() == 4);

      myvec.unset(300);
      REQUIRE(myvec.getNNZ() == 3);

      myvec.setAll(2);
      REQUIRE(myvec.getNNZ() == 1000);

      GraphMat::Clear(&myvec);
      REQUIRE(myvec.getNNZ() == 0);
  }

  SECTION(" DenseSegment basic tests", "DenseSegment basic tests") {
      GraphMat::DenseSegment<int> v1(1000);
      v1.set(1, 1);
      v1.set(10, 1);
      v1.set(200, 1);
      v1.set(300, 1);
      REQUIRE(v1.compute_nnz() == 4);

      v1.unset(300);
      REQUIRE(v1.compute_nnz() == 3);
   }

  SECTION("DenseSegment send/recv tests", "DenseSegment send/recv tests") {
      if(GraphMat::get_global_nrank() % 2 == 0)
      {
        GraphMat::DenseSegment<int> v1(1000);
        std::vector<MPI_Request> requests;

        if(GraphMat::get_global_myrank() % 2 == 1)
        {
          REQUIRE(v1.compute_nnz() == 0);
          v1.recv_nnz(GraphMat::get_global_myrank(),
                      GraphMat::get_global_myrank() - 1,
                      &requests);
          v1.recv_segment(GraphMat::get_global_myrank(),
                          GraphMat::get_global_myrank() - 1,
                          &requests);
        }
        else
        {
          v1.set(1, 1);
          v1.set(10, 2);
          v1.set(200, 3);
          v1.set(300, 4);
          REQUIRE(v1.compute_nnz() == 4);
          v1.compress();
          v1.send_nnz(GraphMat::get_global_myrank(),
                      GraphMat::get_global_myrank() + 1,
                      &requests);
          v1.send_segment(GraphMat::get_global_myrank(),
                          GraphMat::get_global_myrank() + 1,
                          &requests);

        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
        requests.clear();
        v1.decompress();
        REQUIRE(v1.compute_nnz() == 4);
        REQUIRE(v1.get(1) == 1);
        REQUIRE(v1.get(10) == 2);
        REQUIRE(v1.get(200) == 3);
        REQUIRE(v1.get(300) == 4);
      }
   }

  SECTION("DenseSegment send/recv tests serialized", "DenseSegment send/recv tests serialized") {
      if(GraphMat::get_global_nrank() % 2 == 0)
      {
        GraphMat::DenseSegment<sv2> v1(1000);
        std::vector<MPI_Request> requests;

        if(GraphMat::get_global_myrank() % 2 == 1)
        {
          REQUIRE(v1.compute_nnz() == 0);
          v1.recv_nnz(GraphMat::get_global_myrank(),
                      GraphMat::get_global_myrank() - 1,
                      &requests);
          v1.recv_segment(GraphMat::get_global_myrank(),
                          GraphMat::get_global_myrank() - 1,
                          &requests);
        }
        else
        {
          {
            sv2 a;
            a.v.push_back(1);
            v1.set(1, a);
          }
          {
            sv2 a;
            a.v.push_back(2);
            v1.set(10, a);
          }
          {
            sv2 a;
            a.v.push_back(3);
            v1.set(200, a);
          }
          {
            sv2 a;
            a.v.push_back(4);
            v1.set(300, a);
          }
          REQUIRE(v1.compute_nnz() == 4);
          v1.compress();
          v1.send_nnz(GraphMat::get_global_myrank(),
                      GraphMat::get_global_myrank() + 1,
                      &requests);
          v1.send_segment(GraphMat::get_global_myrank(),
                          GraphMat::get_global_myrank() + 1,
                          &requests);

        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
        requests.clear();
        v1.decompress();
        sv2 a;
        a = v1.get(1);
        REQUIRE(a.v.size() == 1);
        REQUIRE(a.v[0] == 1);
        a = v1.get(10);
        REQUIRE(a.v.size() == 1);
        REQUIRE(a.v[0] == 2);
        a = v1.get(200);
        REQUIRE(a.v.size() == 1);
        REQUIRE(a.v[0] == 3);
        a = v1.get(300);
        REQUIRE(a.v.size() == 1);
        REQUIRE(a.v[0] == 4);
        REQUIRE(v1.compute_nnz() == 4);
      }
   }
}

