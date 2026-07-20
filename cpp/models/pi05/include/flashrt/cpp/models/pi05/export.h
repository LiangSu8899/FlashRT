#ifndef FLASHRT_CPP_MODELS_PI05_EXPORT_H
#define FLASHRT_CPP_MODELS_PI05_EXPORT_H

#if defined(_WIN32)
#if defined(FLASHRT_PI05_C_BUILDING_LIBRARY)
#define FLASHRT_PI05_C_API __declspec(dllexport)
#else
#define FLASHRT_PI05_C_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define FLASHRT_PI05_C_API __attribute__((visibility("default")))
#else
#define FLASHRT_PI05_C_API
#endif

#endif  // FLASHRT_CPP_MODELS_PI05_EXPORT_H
