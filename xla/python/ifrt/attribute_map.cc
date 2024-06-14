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

#include "xla/python/ifrt/attribute_map.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/python/ifrt/attribute_map.pb.h"

namespace xla {
namespace ifrt {

absl::StatusOr<AttributeMap> AttributeMap::FromProto(
    const AttributeMapProto& proto) {
  AttributeMap::Map map;
  map.reserve(proto.attributes_size());
  for (const auto& [key, value] : proto.attributes()) {
    switch (value.value_case()) {
      case AttributeMapProto::Value::kStringValue:
        map.insert({key, value.string_value()});
        break;
      case AttributeMapProto::Value::kBoolValue:
        map.insert({key, value.bool_value()});
        break;
      case AttributeMapProto::Value::kInt64Value:
        map.insert({key, value.int64_value()});
        break;
      case AttributeMapProto::Value::kInt64List:
        map.insert(
            {key, std::vector<int64_t>(value.int64_list().values().begin(),
                                       value.int64_list().values().end())});
        break;
      case AttributeMapProto::Value::kFloatValue:
        map.insert({key, value.float_value()});
        break;
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported value type: ", value.value_case()));
    }
  }
  return AttributeMap(std::move(map));
}

AttributeMapProto AttributeMap::ToProto() const {
  AttributeMapProto proto;
  for (const auto& [key, value] : map_) {
    AttributeMapProto::Value value_proto;
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, std::string>) {
            value_proto.set_string_value(value);
          } else if constexpr (std::is_same_v<T, bool>) {
            value_proto.set_bool_value(value);
          } else if constexpr (std::is_same_v<T, int64_t>) {
            value_proto.set_int64_value(value);
          } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            auto* int64_list = value_proto.mutable_int64_list();
            int64_list->mutable_values()->Reserve(value.size());
            for (const auto& value : value) {
              int64_list->add_values(value);
            }
          } else if constexpr (std::is_same_v<T, float>) {
            value_proto.set_float_value(value);
          }
        },
        value);
    proto.mutable_attributes()->insert({key, std::move(value_proto)});
  }
  return proto;
}

std::string AttributeMap::DebugString(size_t max_string_length,
                                      size_t max_int64_list_size) const {
  auto fomatter = [=](std::string* out,
                      const AttributeMap::Map::value_type& key_value) {
    absl::StrAppend(out, key_value.first, "=");
    std::visit(
        [&](const auto& value) {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, std::string>) {
            if (value.size() > max_string_length) {
              absl::StrAppend(out, "\"", value.substr(0, max_string_length),
                              "...\"");
            } else {
              absl::StrAppend(out, "\"", value, "\"");
            }
          } else if constexpr (std::is_same_v<T, bool>) {
            absl::StrAppend(out, value ? "true" : "false");
          } else if constexpr (std::is_same_v<T, int64_t>) {
            absl::StrAppend(out, value);
          } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            if (value.size() > max_int64_list_size) {
              absl::StrAppend(
                  out, "[",
                  absl::StrJoin(value.begin(),
                                value.begin() + max_int64_list_size, ", "),
                  "...]");
            } else {
              absl::StrAppend(out, "[", absl::StrJoin(value, ", "), "]");
            }
          } else if constexpr (std::is_same_v<T, float>) {
            absl::StrAppend(out, value);
          }
        },
        key_value.second);
  };

  return absl::StrCat("AttributeMap([", absl::StrJoin(map_, ", ", fomatter),
                      "])");
}

}  // namespace ifrt
}  // namespace xla
