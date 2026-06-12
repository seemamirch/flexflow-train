#include "utils/containers/get_element_counts.h"
#include "test/utils/doctest/fmt/map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_element_counts") {
    std::vector<int> input = {1, 2, 3, 2, 3, 3, 2, 3};
    std::map<int, int> result = get_element_counts(input);
    std::map<int, int> correct = {{1, 1}, {2, 3}, {3, 4}};
    CHECK(result == correct);
  }
}
