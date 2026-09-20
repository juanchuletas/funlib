#include <funlib/weights/weight_loader.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace flib {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "WeightLoader requires IEEE 754 float32");

std::string stringField(const nlohmann::json &entry, const char *key) {
  if (!entry.contains(key) || !entry.at(key).is_string()) {
    throw std::runtime_error(std::string("Missing or invalid string field: ") + key);
  }
  auto value = entry.at(key).get<std::string>();
  if (value.empty() || value.find('\0') != std::string::npos) {
    throw std::runtime_error(std::string("Empty or invalid field: ") + key);
  }
  return value;
}
} // namespace

WeightLoader::WeightLoader(const std::filesystem::path &manifest,
                           std::optional<sycl::queue> queue) {
  try {
    std::ifstream input(manifest);
    if (!input) {
      throw std::runtime_error("Cannot open manifest");
    }
    const auto document = nlohmann::json::parse(input);
    if (!document.is_object() || !document.contains("weights") ||
        !document.at("weights").is_array()) {
      throw std::runtime_error("Manifest must contain a weights array");
    }
    const auto base = std::filesystem::canonical(
        manifest.has_parent_path() ? manifest.parent_path() : ".");
    for (const auto &entry : document.at("weights")) {
      if (!entry.is_object()) {
        throw std::runtime_error("Each weight must be an object");
      }
      const auto name = stringField(entry, "name");
      try {
        if (contains(name)) {
          throw std::runtime_error("Duplicate weight name");
        }
        if (stringField(entry, "dtype") != "float32") {
          throw std::runtime_error("Only dtype float32 is supported; convert during export");
        }
        if (!entry.contains("shape") || !entry.at("shape").is_array() ||
            entry.at("shape").empty()) {
          throw std::runtime_error("Shape must be a nonempty array; scalar tensors are unsupported");
        }
        std::vector<std::size_t> shape;
        std::size_t count = 1;
        for (const auto &dimension : entry.at("shape")) {
          if (!dimension.is_number_integer() ||
              (!dimension.is_number_unsigned() && dimension.get<std::int64_t>() < 0)) {
            throw std::runtime_error("Shape dimensions must be nonnegative integers");
          }
          const auto value = dimension.get<std::uint64_t>();
          if (value > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("Shape dimension is too large");
          }
          const auto size = static_cast<std::size_t>(value);
          if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) {
            throw std::runtime_error("Shape element count overflow");
          }
          count *= size;
          shape.push_back(size);
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
            count > static_cast<std::uintmax_t>(
                        std::numeric_limits<std::streamsize>::max()) / sizeof(float)) {
          throw std::runtime_error("Tensor byte count overflow");
        }
        const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
        const std::filesystem::path relative(stringField(entry, "file"));
        if (relative.is_absolute() || relative.has_root_path()) {
          throw std::runtime_error("Weight file must be relative to the manifest directory");
        }
        const auto path = std::filesystem::canonical(base / relative);
        const auto within = path.lexically_relative(base);
        if (within.empty() || *within.begin() == ".." ||
            !std::filesystem::is_regular_file(path)) {
          throw std::runtime_error("Weight file must be a regular file within the manifest directory");
        }
        std::ifstream binary(path, std::ios::binary | std::ios::ate);
        if (!binary || binary.tellg() != std::streampos(bytes)) {
          throw std::runtime_error("Missing or incorrectly sized binary: " + path.string());
        }
        binary.seekg(0);
        std::vector<float> values(count);
        if (bytes != 0 &&
            !binary.read(reinterpret_cast<char *>(values.data()), bytes)) {
          throw std::runtime_error("Cannot read binary: " + path.string());
        }
        const std::uint32_t endian_probe = 1;
        if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1) {
          for (float &value : values) {
            std::array<unsigned char, sizeof(float)> buffer;
            std::memcpy(buffer.data(), &value, sizeof(float));
            std::reverse(buffer.begin(), buffer.end());
            std::memcpy(&value, buffer.data(), sizeof(float));
          }
        }
        if (queue) {
          Tensor<float> tensor(shape, *queue);
          if (count != 0) {
            tensor.copy_from(values.data(), *queue).wait_and_throw();
          }
          weights_.emplace(name, std::move(tensor));
        } else {
          weights_.emplace(name, Tensor<float>(shape, values.data()));
        }
      } catch (const std::exception &error) {
        throw std::runtime_error("Weight '" + name + "': " + error.what());
      }
    }
  } catch (const std::exception &error) {
    throw std::runtime_error("WeightLoader '" + manifest.string() + "': " + error.what());
  }
}

const Tensor<float> &WeightLoader::at(const std::string &name) const {
  const auto found = weights_.find(name);
  if (found == weights_.end()) {
    throw std::out_of_range("WeightLoader: unknown weight '" + name + "'");
  }
  return found->second;
}

bool WeightLoader::contains(const std::string &name) const {
  return weights_.find(name) != weights_.end();
}
} // namespace flib
