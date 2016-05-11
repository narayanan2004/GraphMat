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

enum edge_direction {OUT_EDGES, IN_EDGES, ALL_EDGES};
//enum execution_direction {PUSH, PULL};
enum activity_type {ACTIVE_ONLY, ALL_VERTICES};

template <class T, class U, class V, class E=int> //T::message_type, U::message_reduction_type, V::vertex_property_type, E::edge_type
class GraphProgram {
  protected:
    edge_direction order;
//    execution_direction push_or_pull;
    activity_type activity;

  public:

  GraphProgram() {
    order = OUT_EDGES;
//    push_or_pull = PUSH;
    activity = ACTIVE_ONLY;
  }

  edge_direction getOrder() const {
    return order;
  }
  // execution_direction getPushOrPull() const {
    // return push_or_pull;
  // }
  activity_type getActivity() const {
    return activity;
  }

  virtual void reduce_function(U& v, const U& w) const {
    //v += w;
    std::cout << "Trying to use default (null) reduce_function" << std::endl;
    exit(1);
  }

  virtual void process_message(const T& message, const E edge_val, const V& vertexprop, U& res) const {
    //res = message * edge_val;
    std::cout << "Trying to use default (null) process_message " << std::endl;
    exit(1);
  }

  virtual bool send_message(const V& vertexprop, T& message) const {
    //message = (T)vertexprop;
    std::cout << "Trying to use default (null) send_message " << std::endl;
    exit(1);
    return true;
  }

  virtual void apply(const U& message_out, V& vertexprop)  {
    //vertexprop = (V)message_out;
    std::cout << "Trying to use default (null) apply " << std::endl;
    exit(1);
  }

  virtual void do_every_iteration(int iteration_number) {
  }

};

//-------------------------------------------------------------------------
