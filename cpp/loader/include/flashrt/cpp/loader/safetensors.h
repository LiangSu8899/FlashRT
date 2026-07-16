#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace flashrt {
namespace loader {

struct SafetensorInfo {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t data_offset = 0;
    std::uint64_t bytes = 0;
};

class SafetensorsFile {
public:
    SafetensorsFile() = default;
    ~SafetensorsFile();

    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;
    SafetensorsFile(SafetensorsFile&& other) noexcept;
    SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;

    bool open(const std::string& path);
    void close();

    bool is_open() const { return mapping_ != nullptr; }
    const std::string& path() const { return path_; }
    const std::string& error() const { return error_; }
    std::uint64_t file_bytes() const { return mapping_bytes_; }
    std::uint64_t data_offset() const { return data_offset_; }

    const std::map<std::string, SafetensorInfo>& tensors() const {
        return tensors_;
    }
    const std::map<std::string, std::string>& metadata() const {
        return metadata_;
    }
    const SafetensorInfo* find(const std::string& name) const;
    const void* data(const SafetensorInfo& tensor) const;

private:
    void move_from(SafetensorsFile&& other) noexcept;

    int fd_ = -1;
    void* mapping_ = nullptr;
    std::uint64_t mapping_bytes_ = 0;
    std::uint64_t data_offset_ = 0;
    std::string path_;
    std::string error_;
    std::map<std::string, SafetensorInfo> tensors_;
    std::map<std::string, std::string> metadata_;
};

}  // namespace loader
}  // namespace flashrt
