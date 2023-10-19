/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/reduce_scatter_combiner.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/all_reduce_key.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

int64_t FindMostFrequentGatherDim(
    absl::Span<HloInstruction* const> to_combine) {
  assert(!to_combine.empty());

  // Count frequencies.
  std::vector<int64_t> frequency;
  for (const HloInstruction* it : to_combine) {
    const HloInstruction* wrapped = it->opcode() == HloOpcode::kAsyncStart
                                        ? it->async_wrapped_instruction()
                                        : it;
    int64_t dim =
        Cast<HloReduceScatterInstruction>(wrapped)->scatter_dimension();
    frequency.resize(std::max(dim + 1, static_cast<int64_t>(frequency.size())),
                     0);
    frequency[dim]++;
  }

  int64_t most_frequent_dim = std::distance(
      frequency.begin(), std::max_element(frequency.begin(), frequency.end()));
  return most_frequent_dim;
}

using ReduceScatterKey =
    std::tuple<AllReduceKey, /*scatter_dimension*/ int64_t>;

// Combines the elements of to_combine into a single ReduceScatter op. All
// entries in to_combine must be ReduceScatter ops with exactly one operand
// and the same reduction operation.
Status CombineReduceScatters(
    HloModule* module, absl::Span<HloInstruction* const> to_combine,
    absl::Span<HloInstruction* const> to_combine_ends) {
  if (to_combine.size() < 2) {
    return OkStatus();
  }
  VLOG(1) << "Combined " << to_combine.size() << " reduce-scatter ops";

  bool is_async = !to_combine_ends.empty();
  HloComputation& computation = *to_combine.back()->parent();
  const HloInstruction* front = to_combine.front();
  const HloInstruction* wrapped_front =
      is_async ? front->async_wrapped_instruction() : front;
  HloComputation* reduction = wrapped_front->to_apply();
  std::optional<ReductionKind> first_reduction_kind =
      MatchReductionComputation(reduction);
  TF_RET_CHECK(first_reduction_kind);

  // Create a single bigger ReduceScatter of the operands of the smaller
  // ReduceScatters.
  std::vector<HloInstruction*> operands;
  std::vector<std::optional<std::vector<int64_t>>> operand_permutations;
  std::vector<Shape> output_shapes;

  // Find the most frequent all-gather dimension.
  int64_t most_frequent_dim = FindMostFrequentGatherDim(to_combine);

  VLOG(1) << "Combining set";
  for (HloInstruction* wrapper : to_combine) {
    VLOG(1) << "Set element: " << wrapper->ToString();

    TF_RET_CHECK(wrapper->opcode() == (is_async ? HloOpcode::kAsyncStart
                                                : HloOpcode::kReduceScatter));
    HloInstruction* hlo =
        is_async ? wrapper->async_wrapped_instruction() : wrapper;
    const auto* rs = Cast<HloReduceScatterInstruction>(hlo);

    TF_RET_CHECK(hlo->operands().size() == 1);
    std::optional<ReductionKind> reduction_kind =
        MatchReductionComputation(hlo->to_apply());
    TF_RET_CHECK(reduction_kind);
    TF_RET_CHECK(*reduction_kind == *first_reduction_kind);
    TF_RET_CHECK(hlo->shape().IsArray());

    HloInstruction* operand = wrapper->operands().front();
    operands.push_back(operand);
    operand_permutations.emplace_back();
    output_shapes.push_back(hlo->shape());

    // Bitcast operand if needed.
    if (rs->scatter_dimension() != most_frequent_dim) {
      // TODO(ecg): make this work with async combining.
      TF_RET_CHECK(!is_async);

      const Shape& operand_shape = operand->shape();

      // Build permutation to align gather dimension.
      auto& perm = operand_permutations.back();
      perm = std::vector<int64_t>(operand_shape.rank());
      std::iota(perm->begin(), perm->end(), 0);
      std::swap((*perm)[most_frequent_dim], (*perm)[rs->scatter_dimension()]);

      // Bitcast operand and update output shape.
      operands.back() =
          computation.AddInstruction(HloInstruction::CreateBitcast(
              ShapeUtil::PermuteDimensions(*perm, operand_shape), operand));
      output_shapes.back() = ShapeUtil::PermuteDimensions(*perm, hlo->shape());
    }
  }

  // Create combined scatter-reduce op with a tuple result.
  TF_RET_CHECK(operands.size() >= 2);
  HloInstruction* combined_sync =
      computation.AddInstruction(HloInstruction::CreateReduceScatter(
          ShapeUtil::MakeTupleShape(output_shapes), operands, reduction,
          wrapped_front->replica_groups(),
          /*constrain_layout=*/false, wrapped_front->channel_id(),
          Cast<HloReduceScatterInstruction>(wrapped_front)
              ->use_global_device_ids(),
          most_frequent_dim));

  // We have to propagate the sharding manually because Domain instructions are
  // not guaranteed to preserve it for side effecting instructions.
  if (to_combine.front()->has_sharding()) {
    combined_sync->set_sharding(to_combine.front()->sharding());
  }

  HloInstruction* combined = combined_sync;
  HloInstruction* combined_end = nullptr;
  if (is_async) {
    TF_ASSIGN_OR_RETURN(combined_end, computation.CreateAsyncInstructions(
                                          combined_sync, std::vector<Shape>{},
                                          HloInstruction::kMainExecutionThread,
                                          /*replace=*/true));
    combined = combined_end->mutable_operand(0);
    // Update the schedule to include the new async computation; failing to do
    // this would not pass validation checks.
    // We tweak the schedule later for precisely scheduling the new pair.
    TF_RETURN_IF_ERROR(
        module->schedule().Update({HloInstruction::kMainExecutionThread}));
  }
  VLOG(1) << "Replacing with : " << combined->ToString();

  if (is_async) {
    return CombineCollectives(&computation, combined, combined_end, to_combine,
                              to_combine_ends, is_async);
  }

  // Replace all the smaller ReduceScatters with elements of the tuple output
  // of the single bigger ReduceScatter.
  for (int64_t i = 0; i < to_combine.size(); ++i) {
    HloInstruction* replacement = computation.AddInstruction(
        HloInstruction::CreateGetTupleElement(combined, i));
    if (operand_permutations[i]) {
      replacement = computation.AddInstruction(HloInstruction::CreateBitcast(
          ShapeUtil::PermuteDimensions(*operand_permutations[i],
                                       replacement->shape()),
          replacement));
    }
    TF_RETURN_IF_ERROR(
        computation.ReplaceInstruction(to_combine[i], replacement));
  }
  return OkStatus();
}
}  // namespace

ReduceScatterCombiner::ReduceScatterCombiner(int64_t combine_threshold_in_bytes,
                                             int64_t combine_threshold_count,
                                             bool combine_by_dim, bool is_async,
                                             std::string_view async_strategy,
                                             int64_t near_op_threshold)
    : combine_threshold_in_bytes_(combine_threshold_in_bytes),
      combine_threshold_count_(combine_threshold_count),
      combine_by_dim_(combine_by_dim),
      is_async_(is_async),
      async_strategy_(async_strategy == "near" ? kNear : kTrivial),
      near_op_threshold_(near_op_threshold) {}

StatusOr<bool> ReduceScatterCombiner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running " << name() << " with threshold of "
          << combine_threshold_in_bytes_ << " bytes";

  if (combine_threshold_in_bytes_ <= 0 || combine_threshold_count_ <= 0) {
    VLOG(1) << "Skip " << name() << " because the threshold is zero";
    return false;
  }

  if (hlo_query::ContainsLayoutConstrainedCollective(
          *module, HloOpcode::kReduceScatter)) {
    VLOG(1) << "Skip " << name()
            << " because the module contains "
               "reduce-scatter with constrained layouts";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    HloModule* module = computation->parent();
    if (is_async_) {
      TF_RET_CHECK(module->has_schedule());
    }
    TF_ASSIGN_OR_RETURN(auto domain_map, HloDomainMap::Create(computation, ""));

    auto key_fn = [&domain_map, this](const HloInstruction* instruction)
        -> std::optional<ReduceScatterKey> {
      HloOpcode opcode =
          is_async_ ? HloOpcode::kAsyncStart : HloOpcode::kReduceScatter;
      if (instruction->opcode() != opcode) {
        return std::nullopt;
      }
      const HloInstruction* wrapped =
          is_async_ ? instruction->async_wrapped_instruction() : nullptr;
      if (is_async_ && wrapped->opcode() != HloOpcode::kReduceScatter) {
        return std::nullopt;
      }
      auto* rs = DynCast<HloReduceScatterInstruction>(is_async_ ? wrapped
                                                                : instruction);
      std::optional<AllReduceKey> key =
          GetAllReduceKey(instruction, domain_map.get());

      if (!rs || !key) {
        return std::nullopt;
      }
      if (!MatchReductionComputation(rs->to_apply())) {
        return std::nullopt;
      }

      // Ignore dimension (set to -1) if we are not grouping by dimension.
      int64_t rs_dim_key = this->combine_by_dim_ ? rs->scatter_dimension() : -1;
      return ReduceScatterKey{std::move(*key), rs_dim_key};
    };

    auto size_fn =
        [this](const HloInstruction* instruction) -> StatusOr<int64_t> {
      if (!is_async_) {
        return internal::SizeFromArrayShapedInstruction(instruction);
      }
      TF_RET_CHECK(instruction->opcode() == HloOpcode::kAsyncStart);
      TF_RET_CHECK(instruction->async_wrapped_instruction()->opcode() ==
                   HloOpcode::kReduceScatter);
      // We only care about the output shape.
      TF_RET_CHECK(instruction->shape().IsTuple());
      TF_RET_CHECK(instruction->shape().tuple_shapes_size() == 2);
      const Shape& output_shape = instruction->shape().tuple_shapes(1);
      TF_RET_CHECK(output_shape.IsArray());
      return ShapeUtil::ByteSizeOf(output_shape);
    };

    TF_ASSIGN_OR_RETURN(
        bool computation_changed,
        CombineInstructionsByKey<ReduceScatterKey>(
            computation, key_fn, &CombineReduceScatters,
            combine_threshold_in_bytes_, combine_threshold_count_, is_async_,
            async_strategy_, near_op_threshold_, size_fn));
    changed |= computation_changed;
  }

  return changed;
}

}  // namespace xla
