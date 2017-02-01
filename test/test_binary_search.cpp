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
/* Narayanan Sundaram (Intel Corp.)
 * ******************************************************************************/

#include "catch.hpp"
#include "generator.h"
#include <algorithm>
#include <climits>
#include "GraphMatRuntime.h"

TEST_CASE("binary search tests", "binary search test")
{
  SECTION("Binary search right border") {
    int vec[9] = {1,1,2,2,2,5,5,5,5};
    REQUIRE(1 == GraphMat::binary_search_right_border(vec, 1, 0, 9, 9));
    REQUIRE(4 == GraphMat::binary_search_right_border(vec, 2, 0, 9, 9));
    REQUIRE(-1 == GraphMat::binary_search_right_border(vec, 0, 0, 9, 9));
    REQUIRE(8 == GraphMat::binary_search_right_border(vec, 5, 0, 9, 9));
    REQUIRE(-1 == GraphMat::binary_search_right_border(vec, 3, 0, 9, 9));
    REQUIRE(8 == GraphMat::binary_search_right_border(vec, 5, 3, 9, 9));
    REQUIRE(-1 == GraphMat::binary_search_right_border(vec, 5, 9, 9, 9));
  }

  SECTION("Binary search left border") {
    int vec[9] = {1,1,2,2,2,5,5,5,5};
    REQUIRE(0 == GraphMat::binary_search_left_border(vec, 1, 0, 9, 9));
    REQUIRE(2 == GraphMat::binary_search_left_border(vec, 2, 0, 9, 9));
    REQUIRE(-1 == GraphMat::binary_search_left_border(vec, 0, 0, 9, 9));
    REQUIRE(5 == GraphMat::binary_search_left_border(vec, 5, 0, 9, 9));
    REQUIRE(-1 == GraphMat::binary_search_left_border(vec, 3, 0, 9, 9));
    REQUIRE(5 == GraphMat::binary_search_left_border(vec, 5, 3, 9, 9));
    REQUIRE(-1 == GraphMat::binary_search_left_border(vec, 5, 9, 9, 9));
    REQUIRE(-1 == GraphMat::binary_search_left_border(vec, 6, 0, 9, 9));
    REQUIRE(-1 == GraphMat::binary_search_left_border(vec, 6, 0, 8, 8));
  }

  SECTION("Binary search") {
    int vec[9] = {1,1,2,2,2,5,5,5,5};
    REQUIRE(0 == GraphMat::l_binary_search(0, 9, vec, 1));
    REQUIRE(2 == GraphMat::l_binary_search(0, 9, vec, 2));
    REQUIRE(0 == GraphMat::l_binary_search(0, 9, vec, 0));
    REQUIRE(5 == GraphMat::l_binary_search(0, 9, vec, 4));
    REQUIRE(5 == GraphMat::l_binary_search(0, 9, vec, 5));
    REQUIRE(9 == GraphMat::l_binary_search(0, 9, vec, 7));
  }

  SECTION("Linear search") {
    int vec[9] = {1,1,2,2,2,5,5,5,5};
    REQUIRE(0 == GraphMat::l_linear_search(0, 9, vec, 1));
    REQUIRE(2 == GraphMat::l_linear_search(0, 9, vec, 2));
    REQUIRE(0 == GraphMat::l_linear_search(0, 9, vec, 0));
    REQUIRE(5 == GraphMat::l_linear_search(0, 9, vec, 4));
    REQUIRE(5 == GraphMat::l_linear_search(0, 9, vec, 5));
    REQUIRE(9 == GraphMat::l_linear_search(0, 9, vec, 7));
  }

}

