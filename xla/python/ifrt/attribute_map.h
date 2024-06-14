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

#ifndef XLA_PYTHON_IFRT_ATTRIBUTE_MAP_H_
#define XLA_PYTHON_IFRT_ATTRIBUTE_MAP_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/attribute_map.pb.h"

namespace xla {
namespace ifrt {

// Attribute map that contains UTF-8 keys and variant values.
class AttributeMap {
 public:
  // Supported value types for `AttributeMap`. Modeled after
  // `xla::PjRtValueType`.
  using Value =
      std::variant<std::string, bool, int64_t, std::vector<int64_t>, float>;
  using Map = absl::flat_hash_map<std::string, Value>;

  explicit AttributeMap(Map map) : map_(std::move(map)) {}

  const Map& map() const { return map_; }

  // Deserializes `AttributeMapProto` into `AttributeMap`.
  static absl::StatusOr<AttributeMap> FromProto(const AttributeMapProto& proto);

  // Serializes `AttributeMap` into `AttributeMapProto`.
  AttributeMapProto ToProto() const;

  std::string DebugString(size_t max_string_length = 64,
                          size_t max_int64_list_size = 16) const;

 private:
  Map map_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_ATTRIBUTE_MAP_H_
