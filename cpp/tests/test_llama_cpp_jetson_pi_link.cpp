#include "flashrt/providers/llama_cpp/c_api.h"

#include <cstdio>

extern "C" int frt_llama_cpp_jetson_pi_link_check(void);

int main() {
    if (frt_llama_cpp_jetson_pi_link_check() != 0) {
        std::printf("FAIL: FlashRT llama_cpp provider Jetson-PI link check\n");
        return 1;
    }

    std::printf("ok  : FlashRT llama_cpp provider links Jetson-PI llama/mtmd APIs\n");
    return 0;
}
