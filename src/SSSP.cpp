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
#include <cfloat>

#include "GraphMatRuntime.cpp"

//typedef unsigned char distance_type;
typedef unsigned int distance_type;
//typedef double distance_type;
//typedef float distance_type;

distance_type MAX_DIST = std::numeric_limits<distance_type>::max();

class BFSD2 {
  public: 
    distance_type distance;
  public:
    BFSD2() {
      distance = MAX_DIST;
    }
    bool operator != (const BFSD2& p) {
      return (this->distance != p.distance);
    }
    void print() const {
      if (distance < MAX_DIST) {
        std::cout << "distance = " << distance << std::endl;
      } else {
        std::cout << "distance = INF" << std::endl;
      }
    }
};

class SSSPD {
  public: 
    distance_type distance;
    //long long int parent;
    //long long int id;
    int parent;
    int id;

  public:
    SSSPD() {
      distance = MAX_DIST;
      parent = -1;
      id = -1;
    }
    bool operator != (const SSSPD& p) {
      return (this->distance != p.distance);
    }
    void print() {
      if (distance < MAX_DIST) 
        //printf("id %d\t distance %d\t parent %d\n", id, distance, parent);
        //printf("id %d\t distance %.1f\t parent %d\n", id, distance, parent);
        printf("id %d\t distance %d\t parent %d\n", id, distance, parent);
      else
        printf("id %d\t distance INF\t parent --\n", id);

    }
};

class ID_dist {
  public:
    distance_type distance;
    //long long int id;
    int id;
};

template <class edge_type>
class SSSP : public GraphProgram<distance_type, distance_type, BFSD2, edge_type> {

  public:
    //char current_depth;
    
  public:

  SSSP() {
    //current_depth = 1;
    this->order = OUT_EDGES;
    //async = true;
  }

  void reduce_function(distance_type& a, const distance_type& b) const {
    //return std::min(a,b);
    a = (a<=b)?(a):(b);
  }

  void process_message(const distance_type& message, const edge_type edge_val, const BFSD2& vertexprop, distance_type &res) const {
    res = message + edge_val;
  }

  bool send_message(const BFSD2& vertexprop, distance_type& message) const {
    message = vertexprop.distance;
    return true;
    //return (vertexprop.depth == current_depth-1);
  }

  void apply(const distance_type& message_out, BFSD2& vertexprop)  {
    //if (vertexprop.depth == 255) {
    //  vertexprop.depth = current_depth;
    //}
    vertexprop.distance = std::min(vertexprop.distance, message_out);
  }

  void do_every_iteration(int iteration_number) {
    //current_depth++;
  }

};

class SSSPwithParent : public GraphProgram<ID_dist, ID_dist, SSSPD> {

  public:

  SSSPwithParent() {
    this->order = OUT_EDGES;
    //this->order = ALL_EDGES;
  }

  void reduce_function(ID_dist& a, const ID_dist& b) const {
    a = (a.distance<=b.distance)?(a):(b);
  }

  void process_message(const ID_dist& message, const int edge_val, const SSSPD& vertexprop, ID_dist& res) const {
    res = message;
    res.distance += edge_val;
  }

  bool send_message(const SSSPD& vertexprop, ID_dist& message) const {
    message.id = vertexprop.id;
    message.distance = vertexprop.distance;
    return true;
  }

  void apply(const ID_dist& message_out, SSSPD& vertexprop)  {
    //vertexprop.depth = std::min(vertexprop.depth, message_out);
    if (vertexprop.distance > message_out.distance) {
      vertexprop.distance = message_out.distance;
      vertexprop.parent = message_out.id;
    }
  }

  void do_every_iteration(int iteration_number) {
    //current_depth++;
  }

};

extern unsigned long long int edges_traversed;

void reachable_or_not(BFSD2* v, int* output, void* param=nullptr) {
  int reachable = 0;
  if (v->distance < MAX_DIST) {
    reachable = 1;
  } 
  *output = reachable;
}
void add(int a, int b, int *c, void* param=nullptr) {
  *c = a+b;
}

template<class edge_type>
void run_sssp(const char* filename, int nthreads, int v) {
  //__itt_pause();

  Graph<BFSD2, edge_type> G;
  //Graph<SSSPD> G;
  //G.ReadMTX_sort(filename, nthreads*8); //8 nthread pieces of matrix
  G.ReadMTX(filename, nthreads*8); //8 nthread pieces of matrix
  //G.ReadMTX(filename, 240*8); //8 nthread pieces of matrix

  //BFS b;
  //BFS2 b;
  SSSP<edge_type> b;
  //SSSPwithParent b;
  auto tmp_ds = graph_program_init(b, G);

  //for (int v = 0; v < 25; v++) { 
  //G.reset(); 

  //for (int i = 0; i <= G.nvertices; i++) 
  //  G.vertexproperty[i].id = i;

  BFSD2 init; 
  init.distance = 0; 

  BFSD2 inf; 
  //int v = 10;
  //for (int v = 0; v < 25; v++) {
  G.setAllVertexproperty(inf);
  G.setAllInactive();
  //int v = 1758293;
  //G.vertexproperty[v].distance = 0;
  //G.active[v] = true;

  G.setVertexproperty(v, init);
  G.setActive(v);

  struct timeval start, end;
  gettimeofday(&start, 0);

  //__itt_resume();

  run_graph_program(&b, G, -1, &tmp_ds);
  //run_dense_graph_program(b, G, -1);

  //__itt_pause();

  gettimeofday(&end, 0);
  printf("Time = %.3f ms \n", (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3);
  
 
  int reachable_vertices = 0;

  /*for (int i = 1; i <= G.nvertices; i++) {
    if (G.getVertexproperty(i).distance < MAX_DIST) {
      reachable_vertices++;
    }
  }
  printf("Reachable vertices = %d \n", reachable_vertices);
  
  reachable_vertices = 0;*/
  G.applyReduceAllVertices(&reachable_vertices, reachable_or_not, add);

  printf("Reachable vertices = %d \n", reachable_vertices);

  for (int i = 1; i <= std::min((unsigned long long int)25, (unsigned long long int)G.nvertices); i++) {
  //for (int i = 1; i <= G.nvertices; i++) {
    printf("%d : ", i);
    //G.vertexproperty[i].print();
    G.getVertexproperty(i).print();
    /*if (G.vertexproperty[i].distance < MAX_DIST) {
      printf("PATH: ");
      int par = i;
      //while(par != -1) {
      //  printf(" %d ", par);
      //  par = G.vertexproperty[par].parent;
      //}
      printf("\n");
    }*/
    //if (G.vertexproperty[i].depth < MAX_DIST) {
    //  printf("Depth %d : %d \n", i, G.vertexproperty[i].depth);
    //}
    //else {
    //  printf("Depth %d : INF \n", i);
    //}
  }
  //}
  
  graph_program_clear(tmp_ds);
  //}
  //}

  /*FILE* f;
  f = fopen("out", "w");
  for (int i = 1; i <= G.nvertices; i++) {
      fprintf(f, "Depth %d : %d \n", i, G.vertexproperty[i].depth);
  }
  fclose(f);*/

}

int main (int argc, char* argv[]) {

  const char* input_filename = argv[1];

  if (argc < 3) {
    printf("Correct format: %s A.mtx source_vertex (1-based index)\n", argv[0]);
    return 0;
  }


#pragma omp parallel
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
      printf("num threads got: %d\n", nthreads);
    }
  }
  
  int source_vertex = atoi(argv[2]);
  run_sssp<int>(input_filename, nthreads, source_vertex);
 
 
}

