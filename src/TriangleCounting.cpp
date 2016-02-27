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

#include <vector>
#include <set>
#include <algorithm>
#include <assert.h>
#include <memory>
#include "GraphMatRuntime.cpp"

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
    std::vector<int> neighbors;
    int* bitvector;

  public:
    TC() {
      triangles = 0;
      neighbors.clear();
      bitvector = NULL;
    }
    int operator!=(const TC& t) const {
      return (t.triangles != this->triangles); //dummy
    }
    void print() const {
      printf("%d : %d : ", id, triangles);
      printf("\n");
    }
    ~TC() {
      neighbors.clear();
      triangles = 0;
    }
};

class TCP { //pointer to TC
  public:
  const TC* ptr;
};


int compare(const void* a, const void* b) {
  if ( *(int*)a < *(int*)b ) return -1;
  if ( *(int*)a == *(int*)b ) return 0;
  else return 1;
}

class GetNeighbors : public GraphProgram<int, std::vector<int>, TC> {

  public:
      int BVLENGTH;


  GetNeighbors(int maxvertices) {
    this->order = IN_EDGES;
    BVLENGTH = (maxvertices+31)/32 + 32; //32 for safety
  }

  void reduce_function(std::vector<int>& a, const std::vector<int>& b) const {
    assert(b.size() == 1);
    a.push_back(b[0]);
  }

  void process_message(const int& message, const int edge_val, const TC& vertexprop, std::vector<int>&res) const {
    res.clear(); 
	res.push_back(message);
  }
  bool send_message(const TC& vertexprop, int& message) const {
    message = vertexprop.id;
    return true;
  }
  void apply(const std::vector<int>& message_out, TC& vertexprop) {
    vertexprop.neighbors = message_out;

    if (vertexprop.neighbors.size() > 1024) {
      vertexprop.bitvector = new int[BVLENGTH];
      for (auto it = vertexprop.neighbors.begin(); it != vertexprop.neighbors.end(); ++it) {
        set_bit(*it, vertexprop.bitvector);
      }
    } else {
      std::sort(vertexprop.neighbors.begin(), vertexprop.neighbors.end());
    }
  }

};


class CountTriangles: public GraphProgram<TCP, int, TC> {

  public:
    int BVLENGTH;

  CountTriangles(int maxvertices) {
    this->order = OUT_EDGES;
    BVLENGTH = (maxvertices+31)/32 + 32; //32 for safety

  }

  void reduce_function(int& v, const int& w) const {
    v += w;
  }

  void process_message(const TCP& message_ptr, const int edge_val, const TC& vertexprop, int& res) const {
    res = 0;
    const TC& message = *(message_ptr.ptr);
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
    }


  }

  bool send_message(const TC& vertexprop, TCP& message) const {
    message.ptr = &vertexprop;
    return true;
  }

  void apply(const int& message_out, TC& vertexprop) {
    vertexprop.triangles = message_out;
  }

};

void run_triangle_counting(char* filename, int nthreads) {
  Graph<TC> G;
  G.ReadMTX(filename, nthreads*16); //nthread pieces of matrix
  
  int numberOfVertices = G.getNumberOfVertices();
  GetNeighbors gn(numberOfVertices);
  CountTriangles ct(numberOfVertices);

  auto gn_tmp = graph_program_init(gn, G);
  auto ct_tmp = graph_program_init(ct, G);
  
  struct timeval start, end;

  for (int i = 1; i <= numberOfVertices; i++) {
	TC vp = G.getVertexproperty(i);
	vp.id = i;
    G.setVertexproperty(i, vp);
  }
  gettimeofday(&start, 0);


  G.setAllActive();
  run_graph_program(&gn, G, 1, &gn_tmp);

  G.setAllActive();
  run_graph_program(&ct, G, 1, &ct_tmp);
  
  gettimeofday(&end, 0);
  printf("Time = %.3f ms \n", (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3);

  graph_program_clear(gn_tmp);
  graph_program_clear(ct_tmp);  

  unsigned long int ntriangles = 0;
  for (int i = 1; i <= numberOfVertices; i++) ntriangles += G.getVertexproperty(i).triangles;
  printf("Total triangles = %lu \n", ntriangles);
  
  for (int i = 1; i <= std::min(10, numberOfVertices); i++) {
    G.getVertexproperty(i).print();
  }

}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Correct format: %s A.mtx\n", argv[0]);
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
  
  run_triangle_counting(argv[1], nthreads); 

  }

