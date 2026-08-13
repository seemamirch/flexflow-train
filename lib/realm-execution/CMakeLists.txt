# realm_redop_registry.cu needs to be compiled by nvcc (not g++) so that its
# GPU reduction kernels (SumReduction<T>::apply_cuda/fold_cuda) actually get
# generated; see the comment on register_sum_redop for why that matters.
project(realm-execution LANGUAGES CXX CUDA)

ff_add_library(
  NAME
    realm-execution
  SRC_PATTERNS
    src/*.cc
    src/*.cu
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    compiler
    kernels
    local-execution
    op-attrs
    pcg
    deps::spdlog
    task-spec
    utils
    deps::realm
)

set_target_properties(
  realm-execution
  PROPERTIES
  CUDA_STANDARD 17
)

add_subdirectory(test)
