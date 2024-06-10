/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_FFT_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_FFT_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// This class stores everything that is needed to launch an FFT.
// It is generated by IrEmitter.
//
// This is thread-compatible.
class FftThunk final : public Thunk {
 public:
  // Constructs a thunk for launching an FFT on a host.
  // Semantics of null hlo_instruction argument are as in Thunk.
  FftThunk(Info thunk_info, const DebugOptions& debug_options, int32_t fft_type,
           absl::Span<const int64_t> fft_length,
           const BufferAllocation::Slice& input_buffer,
           const BufferAllocation::Slice& output_buffer,
           const Shape& input_shape, const Shape& output_shape);

  static absl::StatusOr<std::unique_ptr<FftThunk>> Create(
      Info thunk_info, const DebugOptions& debug_options, int32_t fft_type,
      absl::Span<const int64_t> fft_length,
      const BufferAllocation::Slice& input_buffer,
      const BufferAllocation::Slice& output_buffer, const Shape& input_shape,
      const Shape& output_shape);

  tsl::AsyncValueRef<Thunk::ExecuteEvent> Execute(
      const ExecuteParams& params) final;

  BufferUses buffer_uses() const final;

 private:
  const DebugOptions& debug_options_;
  const bool double_precision_;
  const int32_t fft_type_;
  const std::vector<int64_t> fft_length_;

  const BufferAllocation::Slice input_buffer_;
  const BufferAllocation::Slice output_buffer_;

  const Shape input_shape_;
  const Shape output_shape_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_FFT_THUNK_H_
