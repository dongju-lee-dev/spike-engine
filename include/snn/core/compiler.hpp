#pragma once

#include <cstddef>

#if defined(__clang__) || defined(__GNUC__)
    #define SNN_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define SNN_RESTRICT __restrict
#else 
    #define SNN_RESTRICT
#endif

#if defined(__clang__) || defined(__GNUC__)
    #define SNN_ASSUME_ALIGNED(ptr, alignment) \
        reinterpret_cast<decltype(ptr)>(__builtin_assume_aligned((ptr), (alignment)))
#else
    #define SNN_ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif