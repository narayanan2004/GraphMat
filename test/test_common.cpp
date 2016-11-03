
#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "src/graphpad.h"

int main(int argc, char * argv[])
{
  MPI_Init(NULL,NULL);
  GraphPad::GB_Init();

  int res =  Catch::Session().run(argc, argv);

  MPI_Finalize();
  return res;
}
