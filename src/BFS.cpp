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

#include <climits>
#include "GraphMatRuntime.cpp"

typedef unsigned int depth_type;

depth_type MAX_DIST = INT_MAX;

class BFSD {
  public: 
    depth_type depth;
    depth_type old_depth;
  public:
    BFSD() {
      depth = 255;//INT_MAX;
      old_depth = 255;//INT_MAX;
    }
    bool operator != (const BFSD& p) {
      return (this->depth != p.depth);
    }
};

class BFS : public GraphProgram<depth_type, depth_type, BFSD> {

  public:
    depth_type current_depth;
    
  public:

  BFS() {
    current_depth = 0;
    this->order = ALL_EDGES;
  }

  void reduce_function(depth_type& a, const depth_type& b) const {
    //return std::min(a,b);
    a = (a<=b)?(a):(b);
  }

  void process_message(const depth_type& message, const int edge_val, const BFSD& vertexprop, depth_type &res) const {
    res = message;
  }

  bool send_message(const BFSD& vertexprop, depth_type& message) const {
    message = vertexprop.depth;
    return (vertexprop.old_depth != vertexprop.depth);
  }

  void apply(const depth_type& message_out, BFSD& vertexprop) {
    vertexprop.old_depth = vertexprop.depth;
    //vertexprop.depth = std::min(message_out+1, vertexprop.depth);
    if (message_out+1 < vertexprop.depth) {
      vertexprop.depth = message_out+1;
    }
  }

};

class BFSD2 {
  public: 
    depth_type depth;
    unsigned long long int parent;
    unsigned long long int id;
  public:
    BFSD2() {
      depth = 255;//INT_MAX;
      parent = -1;
      id = -1;
    }
    bool operator != (const BFSD2& p) {
      return (this->depth != p.depth);
    }
    void print() {
      printf("depth %d \n", depth);
    }
};

//class BFS2 : public GraphProgram<depth_type, depth_type, BFSD2> {
class BFS2 : public GraphProgram<unsigned long long int, unsigned long long int, BFSD2> {

  public:
    depth_type current_depth;
    
  public:

  BFS2() {
    current_depth = 1;
    //this->order = ALL_EDGES;
    this->order = OUT_EDGES;
  }

  //void reduce_function(depth_type& a, const depth_type& b) const {
  void reduce_function(unsigned long long int& a, const unsigned long long int& b) const {
    //return std::min(a,b);
    //a = (a<=b)?(a):(b);
    a=b;
  }

  //void process_message(const depth_type& message, const int edge_val, const BFSD2& vertexprop, depth_type &res) const {
  void process_message(const unsigned long long int& message, const int edge_val, const BFSD2& vertexprop, unsigned long long int &res) const {
    res = message;
  }

  //bool send_message(const BFSD2& vertexprop, depth_type& message) {
  bool send_message(const BFSD2& vertexprop, unsigned long long int& message) const {
    //message = vertexprop.depth;
    message = vertexprop.id;
    return (vertexprop.depth == current_depth-1);
  }

  //void apply(const depth_type& message_out, BFSD2& vertexprop)  {
  void apply(const unsigned long long int& message_out, BFSD2& vertexprop)  {
    if (vertexprop.depth == 255) {
      vertexprop.depth = current_depth;
      vertexprop.parent = message_out;
    }
  }

  void do_every_iteration(int iteration_number) {
    current_depth++;
  }

};

class ID_depth {
public:
  depth_type depth;
  unsigned long long int id;
public:
  ID_depth() {
    depth = 255;
    id = -1;
  }
};

class Calc_Parent : public GraphProgram<ID_depth, ID_depth, BFSD2> {

    
  public:

  Calc_Parent() {
    this->order = ALL_EDGES;
  }

  //void reduce_function(depth_type& a, const depth_type& b) const {
  void reduce_function(ID_depth& a, const ID_depth& b) const {
    //return std::min(a,b);
    //a = (a<=b)?(a):(b);
    a = (a.depth > b.depth)?(b):(a);
  }

  //void process_message(const depth_type& message, const int edge_val, const BFSD2& vertexprop, depth_type &res) const {
  void process_message(const ID_depth& message, const int edge_val, const BFSD2& vertexprop, ID_depth &res) const {
    res = message;
  }

  //bool send_message(const BFSD2& vertexprop, depth_type& message) {
  bool send_message(const BFSD2& vertexprop, ID_depth& message) const {
    //message = vertexprop.depth;
    message.id = vertexprop.id;
    message.depth = vertexprop.depth;
    return true;//(vertexprop.depth == current_depth-1);
  }

  //void apply(const depth_type& message_out, BFSD2& vertexprop)  {
  void apply(const ID_depth& message_out, BFSD2& vertexprop)  {
    if (message_out.depth == vertexprop.depth-1) {
    //  vertexprop.depth = current_depth;
    vertexprop.parent = message_out.id;
    }
  }

  void do_every_iteration(int iteration_number) {
  }

};


void run_bfs(char* filename, int nthreads, int v) {
  //Graph<BFSD> G;
  Graph<BFSD2> G;
  G.ReadMTX(filename, nthreads*4); //nthread pieces of matrix
  
  for (int i = 1; i <= G.nvertices; i++) {
    BFSD2 v;
    v.id = i;
    G.setVertexproperty(i, v);
  }
  //BFS b;
  BFS2 b;
  Calc_Parent pc;

  auto pc_tmp = graph_program_init(pc, G);
  auto b_tmp = graph_program_init(b, G);
  //int v = 1758293;
  //int v = 6;

  //G.vertexproperty[v].depth = 0;
  auto source = G.getVertexproperty(v);
  source.depth = 0;
  G.setVertexproperty(v, source);
  G.setActive(v);

  struct timeval start, end;
  gettimeofday(&start, 0);

  run_graph_program(&b, G, -1, &b_tmp);
  //G.setAllActive();
  //run_graph_program(pc, G, 1, &pc_tmp);
  //run_dense_graph_program(b, G, -1);

  gettimeofday(&end, 0);
  printf("Time = %.3f ms \n", (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3);
 
  graph_program_clear(pc_tmp);
  graph_program_clear(b_tmp);

  int reachable_vertices = 0;
  for (int i = 1; i <= G.nvertices; i++) {
    if (G.getVertexproperty(i).depth < 255) {
      reachable_vertices++;
    }
  }
  printf("Reachable vertices = %d \n", reachable_vertices);

  for (int i = 1; i <= std::min(10, G.nvertices); i++) {
    if (G.getVertexproperty(i).depth < 255) {
      printf("Depth %d : %d \n", i, G.getVertexproperty(i).depth);
    }
    else {
      printf("Depth %d : INF \n", i);
    }
  }

  /*FILE* f;
  f = fopen("out", "w");
  for (int i = 1; i <= G.nvertices; i++) {
      fprintf(f, "Depth %d : %d \n", i, G.vertexproperty[i].depth);
  }
  fclose(f);*/

}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    printf("Correct format: %s A.mtx source_vertex (1-based index)\n", argv[0]);
    return 0;
  }

#ifdef __ASSERT
  printf("\nASSERT***************************************************Asserts are on.*************\n\n");
#endif

  //int NTHREADS = atoi(argv[2]);
  //int tid, nt;

#pragma omp parallel
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
      printf("num threads got: %d\n", nthreads);
    }
  }
  

  int source_vertex = atoi(argv[2]);
  //run_pagerank(argv[1], nthreads);
  run_bfs(argv[1], nthreads, source_vertex);
  //run_triangle_counting(argv[1], nthreads); 
  //run_sgd(argv[1], nthreads); 
  //run_graph_coloring(argv[1], nthreads); 
  
  }

