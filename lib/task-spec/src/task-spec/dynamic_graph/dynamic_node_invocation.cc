#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/set_union.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/optional.h"
#include "utils/containers/values.h"
#include "utils/containers/keys.h"
#include "utils/containers/all_of.h"

namespace FlexFlow {

bool invocation_fully_satisfies(DynamicNodeInvocation const &i,
                                std::function<bool(DynamicNodeAttrs const &)> const &node_condition,
                                std::function<bool(DynamicValueAttrs const &)> const &value_condition,
                                std::function<bool(DynamicTensorSlot const &)> const &slot_condition)
{
  return node_condition(i.node_attrs)
    && all_of(values(i.inputs), value_condition)
    && all_of(keys(i.inputs), slot_condition)
    && all_of(values(i.outputs), value_condition)
    && all_of(keys(i.outputs), slot_condition);
}

void require_invocation_fully_satisfies(DynamicNodeInvocation const &i,
                                        std::function<void(DynamicNodeAttrs const &)> const &require_node_condition,
                                        std::function<void(DynamicValueAttrs const &)> const &require_value_condition,
                                        std::function<void(DynamicTensorSlot const &)> const &require_slot_condition) {
  require_node_condition(i.node_attrs);
  for (DynamicTensorSlot const &k : keys(i.inputs)) {
    require_slot_condition(k);
    require_value_condition(i.inputs.at(k));
  }
  for (DynamicTensorSlot const &k : keys(i.outputs)) {
    require_slot_condition(k);
    require_value_condition(i.outputs.at(k));
  }
}

std::unordered_map<DynamicTensorSlot, DynamicValueAttrs>
    get_slot_map_for_direction(DynamicNodeInvocation const &invocation,
                               TensorDirection direction) {
  switch (direction) {
    case TensorDirection::INCOMING:
      return invocation.inputs;
    case TensorDirection::OUTPUT:
      return invocation.outputs;
    default:
      PANIC("Unexpected direction {}", direction);
  }
}

TrainingOpType
    dynamic_node_invocation_get_op_type(DynamicNodeInvocation const &i) {
  TrainingOperationAttrs training_op_attrs =
      assert_unwrap(i.node_attrs.op_attrs);

  return training_op_attrs_get_op_type(training_op_attrs);
}

std::unordered_set<InternalDynamicSlotSite>
    get_dynamic_slot_sites_for_invocation(DynamicNodeInvocation const &i) {

  std::unordered_set<InternalDynamicSlotSite> input_slots =
      transform(unordered_set_of(i.inputs),
                [&](std::pair<DynamicTensorSlot, DynamicValueAttrs> const &p)
                    -> InternalDynamicSlotSite {
                  return InternalDynamicSlotSite{
                      /*invocation=*/i,
                      /*direction=*/TensorDirection::INCOMING,
                      /*slot_name=*/p.first,
                  };
                });

  std::unordered_set<InternalDynamicSlotSite> output_slots =
      transform(unordered_set_of(i.outputs),
                [&](std::pair<DynamicTensorSlot, DynamicValueAttrs> const &p)
                    -> InternalDynamicSlotSite {
                  return InternalDynamicSlotSite{
                      /*invocation=*/i,
                      /*direction=*/TensorDirection::OUTPUT,
                      /*slot_name=*/p.first,
                  };
                });

  ASSERT(are_disjoint(input_slots, output_slots));

  return set_union(input_slots, output_slots);
}

} // namespace FlexFlow
