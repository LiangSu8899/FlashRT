#ifndef FLASHRT_PROVIDERS_LLAMA_CPP_CHECKPOINT_IDENTITY_H
#define FLASHRT_PROVIDERS_LLAMA_CPP_CHECKPOINT_IDENTITY_H

#include <string>

namespace flashrt::providers::llama_cpp {

bool checkpoint_identity(const char* path, std::string* identity,
                         std::string* error);
void clear_runtime_open_error();
void set_runtime_open_error(const std::string& error);
const char* runtime_open_error();

}  // namespace flashrt::providers::llama_cpp

#endif
