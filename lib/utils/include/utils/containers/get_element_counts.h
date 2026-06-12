#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ELEMENT_COUNTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ELEMENT_COUNTS_H

#include "utils/containers/contains_key.h"
#include <string>
#include <map>
#include <vector>

namespace FlexFlow {

template <typename T>
std::map<T, int> get_element_counts(std::vector<T> const &v) {
  std::map<T, int> counts;
  for (T const &t : v) {
    if (!contains_key(counts, t)) {
      counts[t] = 0;
    }
    counts.at(t)++;
  }
  return counts;
}

std::map<char, int> get_element_counts(std::string const &);

} // namespace FlexFlow

#endif
