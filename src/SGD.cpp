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
#include <omp.h>
#include "GraphMatRuntime.cpp"

const int MAX_THREADS = 120;
unsigned int rseed[16*MAX_THREADS];

template <unsigned int K>
class LatentVector {
  public:
    double lv[K];
    double sqerr;

  public:
    LatentVector() {
      //for (int i = 0; i < K; i++) {
        //lv[i] = 2*((double)rand()/(double)RAND_MAX)-1;
        //lv[i] = ((double)rand()/(double)RAND_MAX);
        //lv[i] = 1.0;//((double)rand()/(double)RAND_MAX);
        //lv[i] = ((double)rand_r(rseed + 16*omp_get_thread_num() )/(double)RAND_MAX);
      //}
    }
    ~LatentVector() {
    }

    bool operator !=(const LatentVector<K>& p) {
      bool result = false;
      for (int i = 0; i < K; i++) {
        if (fabs(p.lv[i] - lv[i]) > 1e-7) {
          result = true;
        }
      }
      return result;
    }
    /*LatentVector<K>& operator=(const LatentVector<K>& p) {
      if (this == &p) return *this;

      for (int i = 0; i <K; i++) lv[i] = p.lv[i];
      return *this;
    }*/

    void operator +=(const LatentVector<K> &p) {
      for (int i = 0; i < K; i++) {
        lv[i] += p.lv[i];
      }
    }
    void scale(double s) {
      for (int i =0; i < K; i++) {
        lv[i] *= s;
      }
    }

    void print() const {
      for (int i = 0; i < K; i++) {
        printf(" %.2f ", lv[i]);
      }
    }

};

template <unsigned int K>
class ConstLatentVectorPtr {
  public:
  const LatentVector<K>* lvptr;
};

template<unsigned int K>
//class SGDProgram : public GraphProgram<ConstLatentVectorPtr<K>, LatentVector<K>, LatentVector<K> > {
class SGDProgram : public GraphProgram<LatentVector<K>, LatentVector<K>, LatentVector<K> > {
  public:
    double lambda;
    double step;

  public:
    SGDProgram(double l, double s) {
      lambda = l;
      step = s;
      this->order = ALL_EDGES;// check
      this->activity = ALL_VERTICES;
    }

  void reduce_function(LatentVector<K>& v, const LatentVector<K>& w) const {
    //v += w;
    for (int i = 0; i < K; i++) v.lv[i] += w.lv[i];
  }

  //void process_message(const ConstLatentVectorPtr<K>& message, const int edge_val, 
  void process_message(const LatentVector<K>& message, const int edge_val, 
                        const LatentVector<K>& vertexprop, LatentVector<K>& res) const {
    //res = message * edge_val;
    double estimate = 0;
    for (int i = 0; i < K; i++) {
      estimate += message.lv[i]*vertexprop.lv[i];
    }
    double error = edge_val - estimate;

    for (int i =0; i < K; i++) {
      //res.lv[i] = -lambda*vertexprop.lv[i] + message.lv[i]*error;
      res.lv[i] =  message.lv[i]*error;
    }
  }

  //bool send_message(const LatentVector<K>& vertexprop, ConstLatentVectorPtr<K>& message) {
  bool send_message(const LatentVector<K>& vertexprop, LatentVector<K>& message) const {
    message = vertexprop;
    //message.lvptr = &vertexprop;
    return true;
  }

  void apply(const LatentVector<K>& message_out, LatentVector<K>& vertexprop) {
    for (int i =0; i < K; i++) {
      vertexprop.lv[i] += step*(-lambda*vertexprop.lv[i] +message_out.lv[i]);
    }
  }


};

template<unsigned int K>
class RMSEProgram : public GraphProgram<LatentVector<K>, double, LatentVector<K> > {
  
  public:
  RMSEProgram() {
    this->order = IN_EDGES;
  }

  public:

  void reduce_function(double& v, const double& w) const {
    v += w;
  }

  void process_message(const LatentVector<K>& message, const int edge_val, 
                        const LatentVector<K>& vertexprop, double& res) const {
    //res = message * edge_val;
    double est = 0;
    for (int i = 0; i < K; i++) {
      est += message.lv[i]*vertexprop.lv[i];
    }
    double error = edge_val - est;
    res = error*error;

  }

  bool send_message(const LatentVector<K>& vertexprop, LatentVector<K>& message) const {
    message = vertexprop;
    return true;
  }

  void apply(const double& message_out, LatentVector<K>& vertexprop) {
    vertexprop.sqerr = message_out;
  }
};

/*template<unsigned int K>
class SGDInitProgram : public GraphProgram<bool, bool, LatentVector<K> > {
  
  public:
  SGDInitProgram() {
    this->order = ALL_EDGES;
  }

  public:

  void apply(const bool& message_out, LatentVector<K>& vertexprop) {
    for (int i = 0; i < K; i++) {
      vertexprop.lv[i] = ((double)rand_r(rseed + 16*omp_get_thread_num() )/(double)RAND_MAX);
    }
  }

};*/

template<class V>
void return_sqerr(V* vertexprop, double* out, void* params) {
  *out = vertexprop->sqerr;
}

void run_sgd(char* filename, int nthreads) {
  const int k = 20;
  Graph< LatentVector<k> > G;
  G.ReadMTX(filename, nthreads*8); //nthread pieces of matrix

  srand(0);
  for (int i = 0; i < nthreads; i++) {
    rseed[16*i] = rand();
  }
  double err = 0.0;

  SGDProgram<k> sgdp(0.001, 0.00000035);
  //SGDInitProgram<k> sgdip;
  RMSEProgram<k> rmsep;

  auto sgdp_tmp = graph_program_init(sgdp, G);
  auto rmsep_tmp = graph_program_init(rmsep, G);

  //G.setAllActive();
  //run_graph_program(&sgdip, G);
  for (int i = 1; i <= G.getNumberOfVertices(); i++) {
    LatentVector<k> v;
    v.sqerr = 0.0;
    unsigned int r = i;
    for (int j = 0; j < k; j++) {
      v.lv[j] = ((double)rand_r(&r)/(double)RAND_MAX);
    }
    G.setVertexproperty(i, v);
  }
  
  G.setAllActive();
  run_graph_program(&rmsep, G, 1, &rmsep_tmp);

  for (int i = 1; i <= G.nvertices; i++) err += G.getVertexproperty(i).sqerr;
  printf("RMSE error = %lf per edge \n", sqrt(err/(G.nnz)));

  printf("SGD Init over\n");
  
  struct timeval start, end;

  gettimeofday(&start, 0);

  G.setAllActive();
  run_graph_program(&sgdp, G, 10, &sgdp_tmp);

  /*
  for (int it = 0; it < 10; it ++) {

  G.setAllActive();
  run_graph_program(sgdp, G, 1, &sgdp_tmp);
  
  //G.setAllActive();
  //run_graph_program(rmsep, G, 1, &rmsep_tmp);

  //err = 0.0;
  //for (int i = 1; i <= G.nvertices; i++) err += G.vertexproperty[i].sqerr;
  //printf("RMSE error = %lf per edge \n", sqrt(err/(G.nnz)));
  }
  */

  gettimeofday(&end, 0);
  
  double time = (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3;
  printf("Time = %.3f ms \n", time);

  G.setAllActive();
  run_graph_program(&rmsep, G, 1, &rmsep_tmp);

  graph_program_clear(sgdp_tmp);
  graph_program_clear(rmsep_tmp);

  err = 0.0;
  for (int i = 1; i <= G.nvertices; i++) err += G.getVertexproperty(i).sqerr;
  printf("RMSE error = %lf per edge \n", sqrt(err/(G.nnz)));

  for (int i = 1; i <= std::min(10, G.nvertices); i++) { 
    printf("%d : ", i) ;
    G.getVertexproperty(i).print();
    printf("\n");
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("Correct format: %s A.mtx \n", argv[0]);
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
  

  //run_pagerank(argv[1], nthreads);
  //run_bfs(argv[1], nthreads);
  //run_triangle_counting(argv[1], nthreads); 
  run_sgd(argv[1], nthreads); 
  //run_graph_coloring(argv[1], nthreads); 
  
}

