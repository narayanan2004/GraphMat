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

#include "GraphMatRuntime.cpp"
#include <vector>
#include <set>
#include <algorithm>
#include <assert.h>
#include <memory>
#include <array>

typedef std::array<int, 256> int256;

void set_bit(unsigned int idx, int* bitvec) {
    unsigned int neighbor_id = idx;
    int dword = (neighbor_id >> 5);
    int bit = neighbor_id  & 0x1F;
    unsigned int current_value = bitvec[dword];
    if ( (current_value & (1<<bit)) == 0)
    {
      bitvec[dword] = current_value | (1<<bit);
    }
}

class TC {
  public:
    int id;
    int triangles;
    int256 neighbors_subset;
    //std::vector<int> neighbors;
    //int* bitvector;

  public:
    TC() {
      triangles = 0;
      neighbors_subset.fill(0);
      //neighbors.clear();
      //bitvector = NULL;
    }
    int operator!=(const TC& t) const {
      return (t.triangles != this->triangles); //dummy
    }
    void print() const {
      printf("%d : %d : ", id, triangles);
      printf("\n");
    }
    ~TC() {
      //neighbors.clear();
      triangles = 0;
    }
};

class GetNeighbors : public GraphProgram<int, int256, TC> {

  public:

  GetNeighbors() {
    //this->activity = ALL_VERTICES;
    this->order = IN_EDGES;
    this->process_message_requires_vertexprop = false;
  }

  void reduce_function(int256& a, const int256& b) const {
    //a.push_back(b[0]);
    //assert(b[0] > 0);
    //for (int i = 0; i < 256; i++) {
    //  if (a[i] == -1) {
    //    a[i] = b[0];
    //    break;
    //  }
    //}
    for(int i = 0; i < 256; i++) a[i] |= b[i];
  }

  void process_message(const int& message, const int edge_val, const TC& vertexprop, int256& res) const {
    //res.clear(); 
    //res.push_back(message);
    res.fill(0);
    //res[0] = message;
    set_bit(message%(256*32), res.data());
  }

  bool send_message(const TC& vertexprop, int& message) const {
    message = vertexprop.id;
    return true;
  }

  void apply(const int256& message_out, TC& vertexprop) {
    /*for (int i = 0; i < 256; i++) {
      if (message_out[i] != -1) {
        vertexprop.neighbors.push_back(message_out[i]);
      }
    }
    std::sort(vertexprop.neighbors.begin(), vertexprop.neighbors.end());*/
    vertexprop.neighbors_subset = message_out;

    /*vertexprop.neighbors = message_out;
    if (vertexprop.neighbors.size() > 1024) {
      vertexprop.bitvector = new int[BVLENGTH];
      for (auto it = vertexprop.neighbors.begin(); it != vertexprop.neighbors.end(); ++it) {
        set_bit(*it, vertexprop.bitvector);
      }
    } else {
      std::sort(vertexprop.neighbors.begin(), vertexprop.neighbors.end());
    }*/
  }

  /*void do_every_iteration(int it_num) {
    iteration++;
    G.setAllInactive();
    for (int i = iteration*256+1; i <= std::min((iteration+1)*256,G.getNumberOfVertices()); i++) {
      G.setActive(i);
    }
  }*/

};


class CountTriangles: public GraphProgram<int256, int, TC> {

  public:

  CountTriangles() {
    this->activity = ALL_VERTICES;
    this->order = OUT_EDGES;
  }

  void reduce_function(int& v, const int& w) const {
    v += w;
  }

  void process_message(const int256& message, const int edge_val, const TC& vertexprop, int& res) const {
    res = 0;
    /*const TC& message = *(message_ptr.ptr);
    //assume sorted
    
    if (message.bitvector == NULL && vertexprop.bitvector == NULL) {
      int it1 = 0, it2 = 0;
      int it1_end = message.neighbors.size();
      int it2_end = vertexprop.neighbors.size();

      while (it1 != it1_end && it2 != it2_end){
        if (message.neighbors[it1] == vertexprop.neighbors[it2]) {
          res++;
          ++it1; ++it2;
        } else if (message.neighbors[it1] < vertexprop.neighbors[it2]) {
          ++it1;
        } else {
          ++it2;
        }
      } 
      return;

    } 

    else {
      int const* bv;
      std::vector<int>::const_iterator itb, ite;

      if (message.bitvector != NULL) { 
        bv = message.bitvector; 
        itb = vertexprop.neighbors.begin(); 
        ite = vertexprop.neighbors.end(); 
      } else { 
        bv = vertexprop.bitvector; 
        itb = message.neighbors.begin(); 
        ite = message.neighbors.end(); 
      }
      for (auto it = itb; it != ite; ++it) {
        res += get_bitvector(*it, bv);
      }
    }*/
    
    /*for (int i = 0; i < 256; i++) {
      if (message[i] > 0) {
        //bool b = std::binary_search(vertexprop.neighbors.begin(), vertexprop.neighbors.end(), message[i]);
        for (int j = 0; j < 256; j++) {
          if (message[i] == vertexprop.neighbors_subset[j]) {
            res++;
            break;
          }
        }
      }
    }*/
    for (int i = 0; i < 256; i++) {
      unsigned int r = message[i] & vertexprop.neighbors_subset[i];
      res += __builtin_popcount(r);
    }
    assert(res>=0 and res<=256*32);

  }

  bool send_message(const TC& vertexprop, int256& message) const {
    //message.ptr = &vertexprop;
    /*message.fill(-1);
    int j = 0;
    for (auto i : vertexprop.neighbors) {
      if (i >= iteration*256+1 && i <= (iteration+1)*256) {
        message[j++] = i;
      }
    }*/
    message = vertexprop.neighbors_subset;
    return true;
  }

  void apply(const int& message_out, TC& vertexprop) {
    vertexprop.triangles += message_out;
  }


};

void return_triangles(TC* v, unsigned long int* out, void* params) {
  *out = v->triangles;
}

void reset_neighbors(TC v, TC* mutable_v, void* params) {
  mutable_v->neighbors_subset.fill(0);
}

void run_triangle_counting(char* filename, int nthreads) {
  Graph<TC> G;
  G.ReadMTX(filename, nthreads*4); //nthread pieces of matrix
  
  int numberOfVertices = G.getNumberOfVertices();
  GetNeighbors gn;
  CountTriangles ct;

  auto gn_tmp = graph_program_init(gn, G);
  auto ct_tmp = graph_program_init(ct, G);
  
  struct timeval start, end;

  for (int i = 1; i <= numberOfVertices; i++) {
    TC vp = G.getVertexproperty(i);
    vp.id = i;
    G.setVertexproperty(i, vp);
  }
  gettimeofday(&start, 0);

  for (int it = 0; it < numberOfVertices/(256*32) + 1; it++) {

  int minv = std::min(numberOfVertices, it*256*32+1);
  int maxv = std::min(numberOfVertices, (it+1)*256*32);
  printf("Processing %d to %d vertices\n", minv, maxv);

  G.setAllInactive();
  for (int i = minv; i <= maxv; i++) {
    G.setActive(i);
  }
  G.applyToAllVertices(reset_neighbors);
  run_graph_program(&gn, G, 1, &gn_tmp);

  G.setAllActive();
  run_graph_program(&ct, G, 1, &ct_tmp);
  }
  
  gettimeofday(&end, 0);
  printf("Time = %.3f ms \n", (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3);

  graph_program_clear(gn_tmp);
  graph_program_clear(ct_tmp);  

  unsigned long int ntriangles = 0;
  //for (int i = 1; i <= numberOfVertices; i++) ntriangles += G.getVertexproperty(i).triangles;
  G.applyReduceAllVertices(&ntriangles, return_triangles, AddFn);
  printf("Total triangles = %lu \n", ntriangles);
  
  for (int i = 1; i <= std::min(10, numberOfVertices); i++) {
    if (G.vertexNodeOwner(i)) {
      G.getVertexproperty(i).print();
    }
  }

}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Correct format: %s A.mtx\n", argv[0]);
    return 0;
  }
  MPI_Init(&argc, &argv);
  GraphPad::GB_Init();

#pragma omp parallel
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
      printf("num threads got: %d\n", nthreads);
    }
  }
  
  run_triangle_counting(argv[1], nthreads); 
  MPI_Finalize();
}

