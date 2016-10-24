#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

int blah(int argc)
{
  return argc+1;
}

TEST_CASE("test","test" ) 
{
  REQUIRE( blah(1) == 2 );
}
