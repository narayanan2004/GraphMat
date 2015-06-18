#include "GraphMatRuntime.cpp"

#include "Degree.cpp"

class PR {
  public:
    double pagerank;
    int degree;
  public:
    PR() {
      pagerank = 0.3;
      degree = 0;
    }
    int operator!=(const PR& p) {
      return (fabs(p.pagerank-pagerank)>1e-8);
    }
};

class PageRank : public GraphProgram<double, double, PR> {
  public:
    double alpha;

  public:

  PageRank(double a=0.3) {
    alpha = a;
    this->activity = ALL_VERTICES;
  }

  void reduce_function(double& a, const double& b) const {
    a += b;
  }
  void process_message(const double& message, const int edge_val, const PR& vertexprop, double& res) const {
    res = message;
  }
  bool send_message(const PR& vertexprop, double& message) const {
    if (vertexprop.degree == 0) {
      message = 0.0;
    } else {
      message = vertexprop.pagerank/(double)vertexprop.degree;
    }
    return true;
  }
  void apply(const double& message_out, PR& vertexprop) {
    vertexprop.pagerank = alpha + (1.0-alpha)*message_out; //non-delta update
  }

};


void run_pagerank(const char* filename, int nthreads) {

  Graph<PR> G;
  PageRank pr;
  Degree<PR> dg;

 
  G.ReadMTX(filename, nthreads*4); //nthread pieces of matrix

  auto dg_tmp = graph_program_init(dg, G);

  struct timeval start, end;
  gettimeofday(&start, 0);

  G.setAllActive();
  run_graph_program(&dg, G, 1, &dg_tmp);

  gettimeofday(&end, 0);
  double time = (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3;
  printf("Degree Time = %.3f ms \n", time);

  graph_program_clear(dg_tmp);
  
  auto pr_tmp = graph_program_init(pr, G);

  gettimeofday(&start, 0);

  G.setAllActive();
  run_graph_program(&pr, G, -1, &pr_tmp);
  
  gettimeofday(&end, 0);
  time = (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3;
  printf("PR Time = %.3f ms \n", time);

  graph_program_clear(pr_tmp);

  for (int i = 0; i < std::min((unsigned long long int)25, (unsigned long long int)G.getNumberOfVertices()); i++) { 
    printf("%d : %d %f\n", i, G.getVertexproperty(i).degree, G.getVertexproperty(i).pagerank);
  }
}

int main(int argc, char* argv[]) {

  const char* input_filename = argv[1];

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
  
  run_pagerank(input_filename, nthreads);

  
}

