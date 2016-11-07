#include "catch.hpp"
#include "generator.hpp"
#include <algorithm>
#include <climits>
#include "GraphMatRuntime.cpp"

class custom_vertex_type {
  public: 
    int  iprop;
    float fprop;
  public:
    custom_vertex_type() {
      iprop = 0;
      fprop = 0.0f;
    }
  friend std::ostream &operator<<(std::ostream &outstream, const custom_vertex_type & val)
    {
      outstream << val.iprop << val.fprop; 
      return outstream;
    }
};



void test_graph(int n) {
  auto E = generate_random_edgelist<int>(n, 16);
  Graph<custom_vertex_type> G;
  G.MTXFromEdgelist(E);

  REQUIRE(G.getNumberOfVertices() == n);

  for (int i = 1; i <= n; i++) {
    if (G.vertexNodeOwner(i)) {
      custom_vertex_type v;
      v.iprop = i;
      v.fprop = i*2.5f;
      G.setVertexproperty(i, v);
    }
  }

  for (int i = 1; i <= n; i++) {
    if (G.vertexNodeOwner(i)) 
      REQUIRE(G.getVertexproperty(i).iprop == i);
      REQUIRE(G.getVertexproperty(i).fprop == Approx(2.5f*i));
  }

}


TEST_CASE("Graph tests", "[random]")
{
  SECTION("size 500") {
    test_graph(500);
  }
  SECTION("size 1000") {
    test_graph(1000);
  }
}
