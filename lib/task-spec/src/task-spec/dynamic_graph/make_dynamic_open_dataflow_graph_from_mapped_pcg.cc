#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mapped_pcg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_tensor_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_layer_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/get_only.h"
#include <optional>
#include <unordered_map>
#include <utility>

namespace FlexFlow {

static bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
get_input_mapping_for_replicate(MappedParallelComputationGraph const &mpcg,
                                parallel_layer_guid_t const &replicate_layer) {

  auto [input_slot_name, input_tensor_guid] =
      get_only(get_incoming_tensors(mpcg.pcg, replicate_layer));

  // find the layer that produces this tensor
  for (auto const &[layer, _] : get_parallel_layer_attrs_mapping(mpcg.pcg)) {
    for (auto const &[slot_name, t] : get_outgoing_tensors(mpcg.pcg, layer)) {
      if (t == input_tensor_guid) {
        MappedOperatorTaskGroup producer_mapping = mpcg.mapped_tasks.at(layer);
        return get_tensor_bindings_for_slot_name(producer_mapping, slot_name);
      }
    }
  }

  PANIC("could not find producer of replicate layer input tensor");
}

static std::unordered_map<parallel_layer_guid_t, TensorSlotName>
get_consumers_of_tensor(MappedParallelComputationGraph const &mpcg,
                        parallel_tensor_guid_t const &tensor) {
  std::unordered_map<parallel_layer_guid_t, TensorSlotName> result;
  for (auto const &[layer, _] : get_parallel_layer_attrs_mapping(mpcg.pcg)) {
    for (auto const &[slot_name, t] : get_incoming_tensors(mpcg.pcg, layer)) {
      if (t == tensor) {
        result.insert({layer, slot_name});
      }
    }
  }
  return result;
}
static std::unordered_map<ParallelTensorSpaceCoordinate,
                          std::unordered_set<MachineSpaceCoordinate>>
build_replicated_output_mapping(MappedParallelComputationGraph const &mpcg,
                                parallel_layer_guid_t const &replicate_layer) {

  // get the output tensor of the replicate layer
  auto [output_slot, output_tensor] =
      get_only(get_outgoing_tensors(mpcg.pcg, replicate_layer));

  // get all successor layers that consume this tensor
  std::unordered_map<ParallelTensorSpaceCoordinate,
                     std::unordered_set<MachineSpaceCoordinate>>
      result;

  for (auto const &[consumer_layer, slot_name] :
       get_consumers_of_tensor(mpcg, output_tensor)) {
    MappedOperatorTaskGroup consumer_mapping =
        mpcg.mapped_tasks.at(consumer_layer);
    bidict<ParallelTensorSpaceCoordinate, MachineSpaceCoordinate>
        tensor_binding =
            get_tensor_bindings_for_slot_name(consumer_mapping, slot_name);
    for (auto const &[p, m] : tensor_binding) {
      result[p].insert(m);
    }
  }
  return result;
}

static DynamicNodeInvocation
build_replicate_invocation(parallel_layer_guid_t const &layer,
                           ParallelLayerAttrs const &attrs,
                           MappedParallelComputationGraph const &mpcg) {

  ReplicateAttrs replicate_attrs = attrs.op_attrs.get<ReplicateAttrs>();

  // input tensor — normal bidict mapping, one machine per coord
  auto input_tensor = get_only(get_incoming_tensors(mpcg.pcg, layer));
  ParallelTensorAttrs input_attrs =
      get_parallel_tensor_attrs(mpcg.pcg, input_tensor.second);

  DynamicValueAttrs input_value{
      /*tensor_guid=*/dynamic_tensor_guid_t{input_tensor.second},
      /*parallel_tensor_shape=*/input_attrs.shape,
      /*shard_coord=*/std::nullopt,
      /*mapping=*/get_input_mapping_for_replicate(mpcg, layer), // bidict p→m
      /*mapping_parallel=*/std::nullopt,
      /*accessor=*/std::nullopt,
      /*role=*/std::nullopt,
  };

  // output tensor — mapping_parallel set, mapping nullopt
  auto output_tensor = get_only(get_outgoing_tensors(mpcg.pcg, layer));
  ParallelTensorAttrs output_attrs =
      get_parallel_tensor_attrs(mpcg.pcg, output_tensor.second);
  // SM replicate_attrs.replicate_degree),  // p→{m0,m1,...}
  DynamicValueAttrs output_value{
      /*tensor_guid=*/dynamic_tensor_guid_t{output_tensor.second},
      /*parallel_tensor_shape=*/output_attrs.shape,
      /*shard_coord=*/std::nullopt,
      /*mapping=*/std::nullopt,
      /*mapping_parallel=*/
      build_replicated_output_mapping(mpcg, layer), // p→{m0,m1,...}
      /*accessor=*/std::nullopt,
      /*role=*/std::nullopt,
  };

  DynamicNodeAttrs node_attrs{
      /*task_type=*/std::nullopt,
      /*device_coord=*/std::nullopt,
      /*mapping=*/std::nullopt, // no MappedOperatorTaskGroup needed
      /*op_attrs=*/TrainingOperationAttrs{attrs.op_attrs},
      /*pcg_layer_guid=*/dynamic_layer_guid_t{layer},
      /*per_device_op_state=*/std::nullopt,
  };

  return DynamicNodeInvocation{
      /*inputs=*/{
          {DynamicTensorSlot{input_tensor.first, std::nullopt}, input_value}},
      /*node_attrs=*/node_attrs,
      /*outputs=*/
      {{DynamicTensorSlot{output_tensor.first, std::nullopt}, output_value}},
  };
}

DynamicOpenDataflowGraph make_dynamic_open_dataflow_graph_from_mapped_pcg(
    MappedParallelComputationGraph const &mpcg) {
  DynamicOpenDataflowGraph result = make_empty_dynamic_open_dataflow_graph();

  for (auto const &[layer, attrs] :
       get_parallel_layer_attrs_mapping(mpcg.pcg)) {

    if (attrs.op_attrs.has<ReplicateAttrs>()) {
      // build replicate invocation directly without going through
      // MappedOperatorTaskGroup — set mapping_parallel on output tensor
      result.invocations.emplace(
          build_replicate_invocation(layer, attrs, mpcg));
      continue;
    }
    DynamicNodeAttrs result_attrs{
        /*task_type=*/std::nullopt,
        /*device_coord=*/std::nullopt,
        /*mapping=*/mpcg.mapped_tasks.at(layer),
        /*op_attrs=*/TrainingOperationAttrs{attrs.op_attrs},
        /*pcg_layer_guid=*/dynamic_layer_guid_t{layer},
        /*per_device_op_state=*/std::nullopt,
    };

    std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> result_inputs =
        transform(get_incoming_tensors(mpcg.pcg, layer),
                  [&](TensorSlotName const &slot_name,
                      parallel_tensor_guid_t const &tensor) {
                    ParallelTensorAttrs attrs =
                        get_parallel_tensor_attrs(mpcg.pcg, tensor);
                    return std::pair<DynamicTensorSlot, DynamicValueAttrs>{
                        DynamicTensorSlot{
                            /*slot_name=*/slot_name,
                            /*slot_tensor_role=*/std::nullopt,
                        },
                        DynamicValueAttrs{
                            /*tensor_guid=*/dynamic_tensor_guid_t{tensor},
                            /*parallel_tensor_shape=*/attrs.shape,
                            /*shard_coord=*/std::nullopt,
                            /*mapping=*/std::nullopt,
                            /*mapping_parallel=*/std::nullopt,
                            /*accessor=*/std::nullopt,
                            /*role=*/std::nullopt,
                        },
                    };
                  });
    std::unordered_map<DynamicTensorSlot, DynamicValueAttrs> result_outputs =
        transform(get_outgoing_tensors(mpcg.pcg, layer),
                  [&](TensorSlotName const &slot_name,
                      parallel_tensor_guid_t const &tensor) {
                    ParallelTensorAttrs attrs =
                        get_parallel_tensor_attrs(mpcg.pcg, tensor);
                    return std::pair<DynamicTensorSlot, DynamicValueAttrs>{
                        DynamicTensorSlot{
                            /*slot_name=*/slot_name,
                            /*slot_tensor_role=*/std::nullopt,
                        },
                        DynamicValueAttrs{
                            /*tensor_guid=*/dynamic_tensor_guid_t{tensor},
                            /*parallel_tensor_shape=*/attrs.shape,
                            /*shard_coord=*/std::nullopt,
                            /*mapping=*/std::nullopt,
                            /*mapping_parallel=*/std::nullopt,
                            /*accessor=*/std::nullopt,
                            /*role=*/std::nullopt,
                        },
                    };
                  });

    result.invocations.emplace(result_inputs, result_attrs, result_outputs);
  }

  return result;
}

} // namespace FlexFlow
