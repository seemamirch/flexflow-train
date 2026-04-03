#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/bidict/algorithms/filter_keys.h"
#include "utils/containers/get_only.h"
#include "utils/containers/map_values2.h"
#include "utils/containers/require_same.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"

namespace FlexFlow {

bool node_is_shard_expanded(DynamicNodeAttrs const &n) {
  return n.device_coord.has_value();
}

bool value_is_shard_expanded(DynamicValueAttrs const &n) {
  return n.shard_coord.has_value();
}

bool no_part_of_graph_is_shard_expanded(DynamicOpenDataflowGraph const &g) {
  auto slot_is_shard_expanded = [](DynamicTensorSlot const &) -> bool {
    return false;
  };

  return no_part_of_dynamic_graph_satisfies(g,
                                            node_is_shard_expanded,
                                            value_is_shard_expanded,
                                            slot_is_shard_expanded);
}

bool graph_is_fully_shard_expanded(DynamicOpenDataflowGraph const &g) {
  auto slot_is_shard_expanded = [](DynamicTensorSlot const &) -> bool {
    return true;
  };

  return full_dynamic_graph_satisfies(g,
                                      node_is_shard_expanded,
                                      value_is_shard_expanded,
                                      slot_is_shard_expanded);
}

// returns (p -> {m0,m1,m2 ...}) for parallel ops
static std::unordered_map<ParallelTensorSpaceCoordinate,
                          std::unordered_set<MachineSpaceCoordinate>>
subset_tensor_mapping_for_coord_replicate(
    std::unordered_map<ParallelTensorSpaceCoordinate,
                       std::unordered_set<MachineSpaceCoordinate>> const
        &mapping,
    ParallelTensorSpaceCoordinate const &coord) {
  return std::unordered_map<ParallelTensorSpaceCoordinate,
                            std::unordered_set<MachineSpaceCoordinate>>{
      {coord, mapping.at(coord)} // preserve all machines for this coord
  };
}

static bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
    restrict_tensor_mapping_keys_to_coord(
        bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> const
            &mapping,
        ParallelTensorSpaceCoordinate const &parallel_tensor_coord) {
  return filter_keys(mapping, [&](ParallelTensorSpaceCoordinate const &p) {
    return p == parallel_tensor_coord;
  });
}

static DynamicNodeInvocation shard_invocation_for_binding_replicate(
    DynamicNodeInvocation const &i, MachineSpaceCoordinate const &dst_machine,
    MachineSpaceCoordinate const &src_machine,
    OperatorAtomicTaskShardBinding const &binding) {

  // Input tensor: normal bidict mapping (one machine per parallel coord)
  auto shard_expand_input_attrs =
      [&](DynamicTensorSlot const &s,
          DynamicValueAttrs const &v) -> DynamicValueAttrs {
    ParallelTensorSpaceCoordinate parallel_tensor_coord =
        binding.tensor_coords.at(s.slot_name);

    DynamicValueAttrs result = v;
    result.shard_coord = parallel_tensor_coord;
    // SM    result.src_machine = src_machine;  // where to read from
    result.mapping = transform(
        v.mapping, [&](bidict<ParallelTensorSpaceCoordinate,
                              MachineSpaceCoordinate> const &mapping) {
          return restrict_tensor_mapping_keys_to_coord(mapping,
                                                       parallel_tensor_coord);
        });
    return result;
  };

  // Output tensor: multimap mapping (many machines per parallel coord)
  auto shard_expand_output_attrs =
      [&](DynamicTensorSlot const &s,
          DynamicValueAttrs const &v) -> DynamicValueAttrs {
    ParallelTensorSpaceCoordinate parallel_tensor_coord =
        binding.tensor_coords.at(s.slot_name);

    DynamicValueAttrs result = v;
    result.shard_coord = parallel_tensor_coord;
    result.mapping_parallel = transform(
        v.mapping_parallel,
        [&](std::unordered_map<ParallelTensorSpaceCoordinate,
                               std::unordered_set<MachineSpaceCoordinate>> const
                &m) {
          return std::unordered_map<ParallelTensorSpaceCoordinate,
                                    std::unordered_set<MachineSpaceCoordinate>>{
              {parallel_tensor_coord, {dst_machine}}};
        });
    return result;
  };

  DynamicNodeAttrs expanded_node_attrs = [&]() {
    DynamicNodeAttrs result = i.node_attrs;
    result.device_coord =
        dst_machine; // this invocation runs on the destination
    return result;
  }();

  return DynamicNodeInvocation{
      /*inputs=*/map_values2(i.inputs, shard_expand_input_attrs),
      /*node_attrs=*/expanded_node_attrs,
      /*outputs=*/map_values2(i.outputs, shard_expand_output_attrs),
  };
}

static DynamicNodeInvocation shard_invocation_for_binding(
    DynamicNodeInvocation const &i,
    MachineSpaceCoordinate const &machine_coord,
    OperatorAtomicTaskShardBinding const &binding) {
  auto shard_expand_value_attrs =
      [&](DynamicTensorSlot const &s,
          DynamicValueAttrs const &v) -> DynamicValueAttrs {
    ParallelTensorSpaceCoordinate parallel_tensor_coord =
        binding.tensor_coords.at(s.slot_name);

    DynamicValueAttrs result = v;
    result.shard_coord = parallel_tensor_coord;
    result.mapping = transform(
        v.mapping,
        [&](bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> const
                &mapping) {
          return restrict_tensor_mapping_keys_to_coord(mapping,
                                                       parallel_tensor_coord);
        });
    return result;
  };

  DynamicNodeAttrs expanded_node_attrs = [&]() {
    DynamicNodeAttrs result = i.node_attrs;
    result.device_coord = machine_coord;
    return result;
  }();

  return DynamicNodeInvocation{
      /*inputs=*/map_values2(i.inputs, shard_expand_value_attrs),
      /*node_attrs=*/expanded_node_attrs,
      /*outputs=*/map_values2(i.outputs, shard_expand_value_attrs),
  };
}

static std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_copy(DynamicNodeInvocation const &i) {
  auto [input_slot, input] = get_only(i.inputs);
  auto [output_slot, output] = get_only(i.outputs);
  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> input_mapping =
      assert_unwrap(input.mapping);
  require_same(input_mapping.left_values(),
               assert_unwrap(output.mapping).left_values());

  return transform(
      input_mapping.left_values(), [&](ParallelTensorSpaceCoordinate const &p) {
        // The machine coord for a copy is inherently nebulous because it
        // doesn't strictly run in any single location. Further, Realm has the
        // flexibility to issue a copy operation from anywhere in the machine,
        // including remotely. Here we choose machine_coord based on the input
        // because we expect this to align with the most efficient way to issue
        // copies in Realm, although the current Realm backend uses a
        // centralized controller and thus issues copies all from a single node.
        MachineSpaceCoordinate machine_coord = input_mapping.at_l(p);

        return shard_invocation_for_binding(i,
                                            machine_coord,
                                            OperatorAtomicTaskShardBinding{{
                                                {input_slot.slot_name, p},
                                                {output_slot.slot_name, p},
                                            }});
      });
}

// SM: replicate op mapping ->
static std::unordered_set<DynamicNodeInvocation>
perform_shard_expansion_for_replicate(DynamicNodeInvocation const &i) {
  auto const &[input_slot, input] = get_only(i.inputs);
  auto const &[output_slot, output] = get_only(i.outputs);

  bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate> input_mapping =
      assert_unwrap(input.mapping);
  std::unordered_map<ParallelTensorSpaceCoordinate,
                     std::unordered_set<MachineSpaceCoordinate>>
      output_mapping = assert_unwrap(output.mapping_parallel);

  require_same(input_mapping.left_values(), keys(output_mapping));
  std::unordered_set<DynamicNodeInvocation> result;
  for (ParallelTensorSpaceCoordinate const &p : input_mapping.left_values()) {
    MachineSpaceCoordinate src_machine = input_mapping.at_l(p);
    // one invocation per destination replica
    for (MachineSpaceCoordinate const &dst_machine : output_mapping.at(p)) {
      result.insert(shard_invocation_for_binding_replicate(
          i,
          dst_machine, // each replica gets its own invocation
          src_machine, // where to read from
          OperatorAtomicTaskShardBinding{{
              {input_slot.slot_name, p},
              {output_slot.slot_name, p},
          }}));
    }
  }
  return result;
}

std::unordered_set<DynamicNodeInvocation>
    perform_shard_expansion_for_invocation(DynamicNodeInvocation const &i) {
  if (i.node_attrs.op_attrs.has_value() &&
      i.node_attrs.op_attrs.value().is_copy()) {
    return perform_shard_expansion_for_copy(i);
  }
  if (i.node_attrs.op_attrs.has_value() &&
      i.node_attrs.op_attrs.value().is_replicate()) {
    return perform_shard_expansion_for_replicate(i);
  }

  MappedOperatorTaskGroup mapping = assert_unwrap(i.node_attrs.mapping);

  std::unordered_set<MachineSpaceCoordinate> shard_machine_coords =
      mapping.get_shard_bindings().left_values();

  return transform(
      shard_machine_coords,
      [&](MachineSpaceCoordinate const &c) -> DynamicNodeInvocation {
        OperatorAtomicTaskShardBinding slot_bindings =
            mapping.get_shard_bindings().at_l(c);

        return shard_invocation_for_binding(i, c, slot_bindings);
      });
}

DynamicOpenDataflowGraph
    perform_shard_expansion(DynamicOpenDataflowGraph const &g) {

  ASSERT(no_part_of_graph_is_shard_expanded(g));

  DynamicOpenDataflowGraph result =
      flatmap_dynamic_invocation_set(g, [&](DynamicNodeInvocation const &i) {
        return perform_shard_expansion_for_invocation(i);
      });

  ASSERT(graph_is_fully_shard_expanded(result));

  return result;
}

} // namespace FlexFlow
