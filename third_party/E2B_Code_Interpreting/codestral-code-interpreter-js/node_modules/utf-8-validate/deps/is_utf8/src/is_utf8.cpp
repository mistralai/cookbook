#ifndef IS_UTF8_H
#define IS_UTF8_H
#include <cstring>

#ifndef IS_UTF8_COMPILER_CHECK_H
#define IS_UTF8_COMPILER_CHECK_H

#ifndef __cplusplus
#error we require a C++ compiler
#endif

#ifndef IS_UTF8_CPLUSPLUS
#if defined(_MSVC_LANG) && !defined(__clang__)
#define IS_UTF8_CPLUSPLUS (_MSC_VER == 1900 ? 201103L : _MSVC_LANG)
#else
#define IS_UTF8_CPLUSPLUS __cplusplus
#endif
#endif

// C++ 17
#if !defined(IS_UTF8_CPLUSPLUS17) && (IS_UTF8_CPLUSPLUS >= 201703L)
#define IS_UTF8_CPLUSPLUS17 1
#endif

// C++ 14
#if !defined(IS_UTF8_CPLUSPLUS14) && (IS_UTF8_CPLUSPLUS >= 201402L)
#define IS_UTF8_CPLUSPLUS14 1
#endif

// C++ 11
#if !defined(IS_UTF8_CPLUSPLUS11) && (IS_UTF8_CPLUSPLUS >= 201103L)
#define IS_UTF8_CPLUSPLUS11 1
#endif

#ifndef IS_UTF8_CPLUSPLUS11
#error we require a compiler compliant with the C++11 standard
#endif

#endif // IS_UTF8_COMPILER_CHECK_H

#ifndef IS_UTF8_COMMON_DEFS_H
#define IS_UTF8_COMMON_DEFS_H

#include <cassert>

#ifndef IS_UTF8_PORTABILITY_H
#define IS_UTF8_PORTABILITY_H

#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#ifndef _WIN32
// strcasecmp, strncasecmp
#include <strings.h>
#endif

#ifdef _MSC_VER
#define IS_UTF8_VISUAL_STUDIO 1
/**
 * We want to differentiate carefully between
 * clang under visual studio and regular visual
 * studio.
 *
 * Under clang for Windows, we enable:
 *  * target pragmas so that part and only part of the
 *     code gets compiled for advanced instructions.
 *
 */
#ifdef __clang__
// clang under visual studio
#define IS_UTF8_CLANG_VISUAL_STUDIO 1
#else
// just regular visual studio (best guess)
#define IS_UTF8_REGULAR_VISUAL_STUDIO 1
#endif // __clang__
#endif // _MSC_VER

#ifdef IS_UTF8_REGULAR_VISUAL_STUDIO
// https://en.wikipedia.org/wiki/C_alternative_tokens
// This header should have no effect, except maybe
// under Visual Studio.
#include <iso646.h>
#endif

#if defined(__x86_64__) || defined(_M_AMD64)
#define IS_UTF8_IS_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define IS_UTF8_IS_ARM64 1
#elif defined(__PPC64__) || defined(_M_PPC64)
//#define IS_UTF8_IS_PPC64 1
#pragma message("The  library does yet support SIMD acceleration under\
POWER processors.")
#else
#define IS_UTF8_IS_32BITS 1

// We do not support 32-bit platforms, but it can be
// handy to identify them.
#if defined(_M_IX86) || defined(__i386__)
#define IS_UTF8_IS_X86_32BITS 1
#elif defined(__arm__) || defined(_M_ARM)
#define IS_UTF8_IS_ARM_32BITS 1
#elif defined(__PPC__) || defined(_M_PPC)
#define IS_UTF8_IS_PPC_32BITS 1
#endif

#endif // defined(__x86_64__) || defined(_M_AMD64)

// this is almost standard?
#define IS_UTF8_STRINGIFY_IMPLEMENTATION_(a) #a
#define IS_UTF8_STRINGIFY(a) IS_UTF8_STRINGIFY_IMPLEMENTATION_(a)

// We are going to use runtime dispatch.
#ifdef IS_UTF8_IS_X86_64
#ifdef __clang__
// clang does not have GCC push pop
// warning: clang attribute push can't be used within a namespace in clang up
// til 8.0 so IS_UTF8_TARGET_REGION and IS_UTF8_UNTARGET_REGION must be
// *outside* of a namespace.
#define IS_UTF8_TARGET_REGION(T)                                               \
  _Pragma(IS_UTF8_STRINGIFY(                                                   \
      clang attribute push(__attribute__((target(T))), apply_to = function)))
#define IS_UTF8_UNTARGET_REGION _Pragma("clang attribute pop")
#elif defined(__GNUC__)
// GCC is easier
#define IS_UTF8_TARGET_REGION(T)                                               \
  _Pragma("GCC push_options") _Pragma(IS_UTF8_STRINGIFY(GCC target(T)))
#define IS_UTF8_UNTARGET_REGION _Pragma("GCC pop_options")
#endif // clang then gcc

#endif // x86

// Default target region macros don't do anything.
#ifndef IS_UTF8_TARGET_REGION
#define IS_UTF8_TARGET_REGION(T)
#define IS_UTF8_UNTARGET_REGION
#endif

#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ >= 11
#define IS_UTF8_GCC11ORMORE 1
#endif //  __GNUC__ >= 11
#endif // defined(__GNUC__) && !defined(__clang__)

#endif // IS_UTF8_PORTABILITY_H
#ifndef IS_UTF8_AVX512_H_
#define IS_UTF8_AVX512_H_

/*
    It's possible to override AVX512 settings with cmake DCMAKE_CXX_FLAGS.

    All preprocessor directives has form `IS_UTF8_HAS_AVX512{feature}`,
    where a feature is a code name for extensions.

    Please see the listing below to find which are supported.
*/

#ifndef IS_UTF8_HAS_AVX512F
#if defined(__AVX512F__) && __AVX512F__ == 1
#define IS_UTF8_HAS_AVX512F 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512DQ
#if defined(__AVX512DQ__) && __AVX512DQ__ == 1
#define IS_UTF8_HAS_AVX512DQ 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512IFMA
#if defined(__AVX512IFMA__) && __AVX512IFMA__ == 1
#define IS_UTF8_HAS_AVX512IFMA 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512CD
#if defined(__AVX512CD__) && __AVX512CD__ == 1
#define IS_UTF8_HAS_AVX512CD 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512BW
#if defined(__AVX512BW__) && __AVX512BW__ == 1
#define IS_UTF8_HAS_AVX512BW 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512VL
#if defined(__AVX512VL__) && __AVX512VL__ == 1
#define IS_UTF8_HAS_AVX512VL 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512VBMI
#if defined(__AVX512VBMI__) && __AVX512VBMI__ == 1
#define IS_UTF8_HAS_AVX512VBMI 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512VBMI2
#if defined(__AVX512VBMI2__) && __AVX512VBMI2__ == 1
#define IS_UTF8_HAS_AVX512VBMI2 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512VNNI
#if defined(__AVX512VNNI__) && __AVX512VNNI__ == 1
#define IS_UTF8_HAS_AVX512VNNI 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512BITALG
#if defined(__AVX512BITALG__) && __AVX512BITALG__ == 1
#define IS_UTF8_HAS_AVX512BITALG 1
#endif
#endif

#ifndef IS_UTF8_HAS_AVX512VPOPCNTDQ
#if defined(__AVX512VPOPCNTDQ__) && __AVX512VPOPCNTDQ__ == 1
#define IS_UTF8_HAS_AVX512VPOPCNTDQ 1
#endif
#endif

#endif // IS_UTF8_AVX512_H_

#if defined(__GNUC__)
// Marks a block with a name so that MCA analysis can see it.
#define IS_UTF8_BEGIN_DEBUG_BLOCK(name)                                        \
  __asm volatile("# LLVM-MCA-BEGIN " #name);
#define IS_UTF8_END_DEBUG_BLOCK(name) __asm volatile("# LLVM-MCA-END " #name);
#define IS_UTF8_DEBUG_BLOCK(name, block)                                       \
  BEGIN_DEBUG_BLOCK(name);                                                     \
  block;                                                                       \
  END_DEBUG_BLOCK(name);
#else
#define IS_UTF8_BEGIN_DEBUG_BLOCK(name)
#define IS_UTF8_END_DEBUG_BLOCK(name)
#define IS_UTF8_DEBUG_BLOCK(name, block)
#endif

// Align to N-byte boundary
#define IS_UTF8_ROUNDUP_N(a, n) (((a) + ((n)-1)) & ~((n)-1))
#define IS_UTF8_ROUNDDOWN_N(a, n) ((a) & ~((n)-1))

#define IS_UTF8_ISALIGNED_N(ptr, n) (((uintptr_t)(ptr) & ((n)-1)) == 0)

#if defined(IS_UTF8_REGULAR_VISUAL_STUDIO)

#define is_utf8_really_inline __forceinline
#define is_utf8_never_inline __declspec(noinline)

#define is_utf8_unused
#define is_utf8_warn_unused

#ifndef is_utf8_likely
#define is_utf8_likely(x) x
#endif
#ifndef is_utf8_unlikely
#define is_utf8_unlikely(x) x
#endif

#define IS_UTF8_PUSH_DISABLE_WARNINGS __pragma(warning(push))
#define IS_UTF8_PUSH_DISABLE_ALL_WARNINGS __pragma(warning(push, 0))
#define IS_UTF8_DISABLE_VS_WARNING(WARNING_NUMBER)                             \
  __pragma(warning(disable : WARNING_NUMBER))
// Get rid of Intellisense-only warnings (Code Analysis)
// Though __has_include is C++17, it is supported in Visual Studio 2017 or
// better (_MSC_VER>=1910).
#ifdef __has_include
#if __has_include(<CppCoreCheck\Warnings.h>)
#include <CppCoreCheck\Warnings.h>
#define IS_UTF8_DISABLE_UNDESIRED_WARNINGS                                     \
  IS_UTF8_DISABLE_VS_WARNING(ALL_CPPCORECHECK_WARNINGS)
#endif
#endif

#ifndef IS_UTF8_DISABLE_UNDESIRED_WARNINGS
#define IS_UTF8_DISABLE_UNDESIRED_WARNINGS
#endif

#define IS_UTF8_DISABLE_DEPRECATED_WARNING IS_UTF8_DISABLE_VS_WARNING(4996)
#define IS_UTF8_DISABLE_STRICT_OVERFLOW_WARNING
#define IS_UTF8_POP_DISABLE_WARNINGS __pragma(warning(pop))

#else // IS_UTF8_REGULAR_VISUAL_STUDIO

#define is_utf8_really_inline inline __attribute__((always_inline))
#define is_utf8_never_inline inline __attribute__((noinline))

#define is_utf8_unused __attribute__((unused))
#define is_utf8_warn_unused __attribute__((warn_unused_result))

#ifndef is_utf8_likely
#define is_utf8_likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef is_utf8_unlikely
#define is_utf8_unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define IS_UTF8_PUSH_DISABLE_WARNINGS _Pragma("GCC diagnostic push")
// gcc doesn't seem to disable all warnings with all and extra, add warnings
// here as necessary
#define IS_UTF8_PUSH_DISABLE_ALL_WARNINGS                                      \
  IS_UTF8_PUSH_DISABLE_WARNINGS                                                \
  IS_UTF8_DISABLE_GCC_WARNING(-Weffc++)                                        \
  IS_UTF8_DISABLE_GCC_WARNING(-Wall)                                           \
  IS_UTF8_DISABLE_GCC_WARNING(-Wconversion)                                    \
  IS_UTF8_DISABLE_GCC_WARNING(-Wextra)                                         \
  IS_UTF8_DISABLE_GCC_WARNING(-Wattributes)                                    \
  IS_UTF8_DISABLE_GCC_WARNING(-Wimplicit-fallthrough)                        \
  IS_UTF8_DISABLE_GCC_WARNING(-Wnon-virtual-dtor)                          \
  IS_UTF8_DISABLE_GCC_WARNING(-Wreturn-type)                                 \
  IS_UTF8_DISABLE_GCC_WARNING(-Wshadow)                                        \
  IS_UTF8_DISABLE_GCC_WARNING(-Wunused-parameter)                            \
  IS_UTF8_DISABLE_GCC_WARNING(-Wunused-variable)
#define IS_UTF8_PRAGMA(P) _Pragma(#P)
#define IS_UTF8_DISABLE_GCC_WARNING(WARNING)                                   \
  IS_UTF8_PRAGMA(GCC diagnostic ignored #WARNING)
#if defined(IS_UTF8_CLANG_VISUAL_STUDIO)
#define IS_UTF8_DISABLE_UNDESIRED_WARNINGS                                     \
  IS_UTF8_DISABLE_GCC_WARNING(-Wmicrosoft-include)
#else
#define IS_UTF8_DISABLE_UNDESIRED_WARNINGS
#endif
#define IS_UTF8_DISABLE_DEPRECATED_WARNING                                     \
  IS_UTF8_DISABLE_GCC_WARNING(-Wdeprecated-declarations)
#define IS_UTF8_DISABLE_STRICT_OVERFLOW_WARNING                                \
  IS_UTF8_DISABLE_GCC_WARNING(-Wstrict-overflow)
#define IS_UTF8_POP_DISABLE_WARNINGS _Pragma("GCC diagnostic pop")

#endif // MSC_VER

#if defined(IS_UTF8_VISUAL_STUDIO)
/**
 * It does not matter here whether you are using
 * the regular visual studio or clang under visual
 * studio.
 */
#if IS_UTF8_USING_LIBRARY
#define IS_UTF8_DLLIMPORTEXPORT __declspec(dllimport)
#else
#define IS_UTF8_DLLIMPORTEXPORT __declspec(dllexport)
#endif
#else
#define IS_UTF8_DLLIMPORTEXPORT
#endif

/// If EXPR is an error, returns it.
#define IS_UTF8_TRY(EXPR)                                                      \
  {                                                                            \
    auto _err = (EXPR);                                                        \
    if (_err) {                                                                \
      return _err;                                                             \
    }                                                                          \
  }

#endif // IS_UTF8_COMMON_DEFS_H
#include <string>

namespace is_utf8_internals {

enum encoding_type {
  UTF8 = 1,      // BOM 0xef 0xbb 0xbf
  UTF16_LE = 2,  // BOM 0xff 0xfe
  UTF16_BE = 4,  // BOM 0xfe 0xff
  UTF32_LE = 8,  // BOM 0xff 0xfe 0x00 0x00
  UTF32_BE = 16, // BOM 0x00 0x00 0xfe 0xff

  unspecified = 0
};

enum endianness { LITTLE, BIG };

std::string to_string(encoding_type bom);

// Note that BOM for UTF8 is discouraged.
namespace BOM {

/**
 * Checks for a BOM. If not, returns unspecified
 * @param input         the string to process
 * @param length        the length of the string in words
 * @return the corresponding encoding
 */

encoding_type check_bom(const uint8_t *byte, size_t length);
encoding_type check_bom(const char *byte, size_t length);
/**
 * Returns the size, in bytes, of the BOM for a given encoding type.
 * Note that UTF8 BOM are discouraged.
 * @param bom         the encoding type
 * @return the size in bytes of the corresponding BOM
 */
size_t bom_byte_size(encoding_type bom);

} // namespace BOM
} // namespace is_utf8_internals
#ifndef ERROR_H
#define ERROR_H
namespace is_utf8_internals {

enum error_code {
  SUCCESS = 0,
  HEADER_BITS, // Any byte must have fewer than 5 header bits.
  TOO_SHORT,   // The leading byte must be followed by N-1 continuation bytes,
             // where N is the UTF-8 character length This is also the error
             // when the input is truncated.
  TOO_LONG, // The leading byte must not be a continuation byte.
  OVERLONG, // The decoded character must be above U+7F for two-byte characters,
            // U+7FF for three-byte characters, and U+FFFF for four-byte
            // characters.
  TOO_LARGE, // The decoded character must be less than or equal to U+10FFFF OR
             // less than or equal than U+7F for ASCII.
  SURROGATE, // The decoded character must be not be in U+D800...DFFF (UTF-8 or
             // UTF-32) OR a high surrogate must be followed by a low surrogate
             // and a low surrogate must be preceded by a high surrogate
             // (UTF-16)
  OTHER // Not related to validation/transcoding.
};

struct result {
  error_code error;
  size_t
      count; // In case of error, indicates the position of the error. In case
             // of success, indicates the number of words validated/written.

  is_utf8_really_inline result();

  is_utf8_really_inline result(error_code, size_t);
};

} // namespace is_utf8_internals
#endif

IS_UTF8_PUSH_DISABLE_WARNINGS
IS_UTF8_DISABLE_UNDESIRED_WARNINGS

#ifndef IS_UTF8_IMPLEMENTATION_H
#define IS_UTF8_IMPLEMENTATION_H
#include <string>
#if !defined(IS_UTF8_NO_THREADS)
#include <atomic>
#endif
#include <tuple>
#include <vector>
/* From
https://github.com/endorno/pytorch/blob/master/torch/lib/TH/generic/simd/simd.h
Highly modified.

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef IS_UTF8_INTERNAL_ISADETECTION_H
#define IS_UTF8_INTERNAL_ISADETECTION_H

#include <cstdint>
#include <cstdlib>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
#include <cpuid.h>
#endif

namespace is_utf8_internals {
namespace internal {

enum instruction_set {
  DEFAULT = 0x0,
  NEON = 0x1,
  AVX2 = 0x4,
  SSE42 = 0x8,
  PCLMULQDQ = 0x10,
  BMI1 = 0x20,
  BMI2 = 0x40,
  ALTIVEC = 0x80,
  AVX512F = 0x100,
  AVX512DQ = 0x200,
  AVX512IFMA = 0x400,
  AVX512PF = 0x800,
  AVX512ER = 0x1000,
  AVX512CD = 0x2000,
  AVX512BW = 0x4000,
  AVX512VL = 0x8000,
  AVX512VBMI2 = 0x10000
};

#if defined(__PPC64__)

static inline uint32_t detect_supported_architectures() {
  return instruction_set::ALTIVEC;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

static inline uint32_t detect_supported_architectures() {
  return instruction_set::NEON;
}

#elif defined(__x86_64__) || defined(_M_AMD64) // x64

namespace {
namespace cpuid_bit {
// Can be found on Intel ISA Reference for CPUID

// EAX = 0x01
constexpr uint32_t pclmulqdq = uint32_t(1)
                               << 1; ///< @private bit  1 of ECX for EAX=0x1
constexpr uint32_t sse42 = uint32_t(1)
                           << 20; ///< @private bit 20 of ECX for EAX=0x1

// EAX = 0x7f (Structured Extended Feature Flags), ECX = 0x00 (Sub-leaf)
// See: "Table 3-8. Information Returned by CPUID Instruction"
namespace ebx {
constexpr uint32_t bmi1 = uint32_t(1) << 3;
constexpr uint32_t avx2 = uint32_t(1) << 5;
constexpr uint32_t bmi2 = uint32_t(1) << 8;
constexpr uint32_t avx512f = uint32_t(1) << 16;
constexpr uint32_t avx512dq = uint32_t(1) << 17;
constexpr uint32_t avx512cd = uint32_t(1) << 28;
constexpr uint32_t avx512bw = uint32_t(1) << 30;
constexpr uint32_t avx512vl = uint32_t(1) << 31;
} // namespace ebx

namespace ecx {
constexpr uint32_t avx512vbmi2 = uint32_t(1) << 6;
} // namespace ecx
} // namespace cpuid_bit
} // namespace

static inline void cpuid(uint32_t *eax, uint32_t *ebx, uint32_t *ecx,
                         uint32_t *edx) {
#if defined(_MSC_VER)
  int cpu_info[4];
  __cpuid(cpu_info, *eax);
  *eax = cpu_info[0];
  *ebx = cpu_info[1];
  *ecx = cpu_info[2];
  *edx = cpu_info[3];
#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
  uint32_t level = *eax;
  __get_cpuid(level, eax, ebx, ecx, edx);
#else
  uint32_t a = *eax, b, c = *ecx, d;
  asm volatile("cpuid\n\t" : "+a"(a), "=b"(b), "+c"(c), "=d"(d));
  *eax = a;
  *ebx = b;
  *ecx = c;
  *edx = d;
#endif
}

static inline uint32_t detect_supported_architectures() {
  uint32_t eax;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
  uint32_t host_isa = 0x0;

  // EBX for EAX=0x1
  eax = 0x1;
  cpuid(&eax, &ebx, &ecx, &edx);

  if (ecx & cpuid_bit::sse42) {
    host_isa |= instruction_set::SSE42;
  }

  if (ecx & cpuid_bit::pclmulqdq) {
    host_isa |= instruction_set::PCLMULQDQ;
  }

  // ECX for EAX=0x7
  eax = 0x7;
  ecx = 0x0; // Sub-leaf = 0
  cpuid(&eax, &ebx, &ecx, &edx);
  if (ebx & cpuid_bit::ebx::avx2) {
    host_isa |= instruction_set::AVX2;
  }
  if (ebx & cpuid_bit::ebx::bmi1) {
    host_isa |= instruction_set::BMI1;
  }
  if (ebx & cpuid_bit::ebx::bmi2) {
    host_isa |= instruction_set::BMI2;
  }
  if (ebx & cpuid_bit::ebx::avx512f) {
    host_isa |= instruction_set::AVX512F;
  }
  if (ebx & cpuid_bit::ebx::avx512bw) {
    host_isa |= instruction_set::AVX512BW;
  }
  if (ebx & cpuid_bit::ebx::avx512cd) {
    host_isa |= instruction_set::AVX512CD;
  }
  if (ebx & cpuid_bit::ebx::avx512dq) {
    host_isa |= instruction_set::AVX512DQ;
  }
  if (ebx & cpuid_bit::ebx::avx512vl) {
    host_isa |= instruction_set::AVX512VL;
  }
  if (ecx & cpuid_bit::ecx::avx512vbmi2) {
    host_isa |= instruction_set::AVX512VBMI2;
  }
  return host_isa;
}
#else // fallback

static inline uint32_t detect_supported_architectures() {
  return instruction_set::DEFAULT;
}

#endif // end SIMD extension detection code

} // namespace internal
} // namespace is_utf8_internals

#endif // IS_UTF8_INTERNAL_ISADETECTION_H

namespace is_utf8_internals {

/**
 * Validate the UTF-8 string. This function may be best when you expect
 * the input to be almost always valid. Otherwise, consider using
 * validate_utf8_with_errors.
 *
 * Overridden by each implementation.
 *
 * @param buf the UTF-8 string to validate.
 * @param len the length of the string in bytes.
 * @return true if and only if the string is valid UTF-8.
 */
bool validate_utf8(const char *buf, size_t len) noexcept;

class implementation {
public:
  virtual const std::string &name() const { return _name; }

  virtual const std::string &description() const { return _description; }

  bool supported_by_runtime_system() const;

  virtual uint32_t required_instruction_sets() const {
    return _required_instruction_sets;
  }

  /**
   * Validate the UTF-8 string.
   *
   * Overridden by each implementation.
   *
   * @param buf the UTF-8 string to validate.
   * @param len the length of the string in bytes.
   * @return true if and only if the string is valid UTF-8.
   */
  is_utf8_warn_unused virtual bool validate_utf8(const char *buf,
                                                 size_t len) const noexcept = 0;

protected:
  /** @private Construct an implementation with the given name and description.
   * For subclasses. */
  is_utf8_really_inline implementation(std::string name,
                                       std::string description,
                                       uint32_t required_instruction_sets)
      : _name(name), _description(description),
        _required_instruction_sets(required_instruction_sets) {}
  virtual ~implementation() = default;

private:
  /**
   * The name of this implementation.
   */
  const std::string _name;

  /**
   * The description of this implementation.
   */
  const std::string _description;

  /**
   * Instruction sets required for this implementation.
   */
  const uint32_t _required_instruction_sets;
};

/** @private */
namespace internal {

class available_implementation_list {
public:
  /** Get the list of available implementations */
  is_utf8_really_inline available_implementation_list() {}
  /** Number of implementations */
  size_t size() const noexcept;
  /** STL const begin() iterator */
  const implementation *const *begin() const noexcept;
  /** STL const end() iterator */
  const implementation *const *end() const noexcept;

  /**
   * Get the implementation with the given name.
   *
   * Case sensitive.
   *
   *     const implementation *impl =
   * is_utf8_internals::available_implementations["westmere"]; if (!impl) {
   * exit(1); } if (!imp->supported_by_runtime_system()) { exit(1); }
   *     is_utf8_internals::active_implementation = impl;
   *
   * @param name the implementation to find, e.g. "westmere", "haswell", "arm64"
   * @return the implementation, or nullptr if the parse failed.
   */
  const implementation *operator[](const std::string &name) const noexcept {
    for (const implementation *impl : *this) {
      if (impl->name() == name) {
        return impl;
      }
    }
    return nullptr;
  }

  /**
   * Detect the most advanced implementation supported by the current host.
   *
   * This is used to initialize the implementation on startup.
   *
   *     const implementation *impl =
   * is_utf8_internals::available_implementation::detect_best_supported();
   *     is_utf8_internals::active_implementation = impl;
   *
   * @return the most advanced supported implementation for the current host, or
   * an implementation that returns UNSUPPORTED_ARCHITECTURE if there is no
   * supported implementation. Will never return nullptr.
   */
  const implementation *detect_best_supported() const noexcept;
};

template <typename T> class atomic_ptr {
public:
  atomic_ptr(T *_ptr) : ptr{_ptr} {}

#if defined(IS_UTF8_NO_THREADS)
  operator const T *() const { return ptr; }
  const T &operator*() const { return *ptr; }
  const T *operator->() const { return ptr; }

  operator T *() { return ptr; }
  T &operator*() { return *ptr; }
  T *operator->() { return ptr; }
  atomic_ptr &operator=(T *_ptr) {
    ptr = _ptr;
    return *this;
  }

#else
  operator const T *() const { return ptr.load(); }
  const T &operator*() const { return *ptr; }
  const T *operator->() const { return ptr.load(); }

  operator T *() { return ptr.load(); }
  T &operator*() { return *ptr; }
  T *operator->() { return ptr.load(); }
  atomic_ptr &operator=(T *_ptr) {
    ptr = _ptr;
    return *this;
  }

#endif

private:
#if defined(IS_UTF8_NO_THREADS)
  T *ptr;
#else
  std::atomic<T *> ptr;
#endif
};

} // namespace internal

/**
 * The list of available implementations compiled into simdutf.
 */
extern IS_UTF8_DLLIMPORTEXPORT const internal::available_implementation_list& get_available_implementations();

/**
 * The active implementation.
 *
 * Automatically initialized on first use to the most advanced implementation
 * supported by this hardware.
 */
extern IS_UTF8_DLLIMPORTEXPORT internal::atomic_ptr<const implementation>& get_active_implementation();

} // namespace is_utf8_internals

#endif // IS_UTF8_IMPLEMENTATION_H

// Implementation-internal files (must be included before the implementations
// themselves, to keep amalgamation working--otherwise, the first time a file is
// included, it might be put inside the #ifdef
// IS_UTF8_IMPLEMENTATION_ARM64/FALLBACK/etc., which means the other
// implementations can't compile unless that implementation is turned on).

IS_UTF8_POP_DISABLE_WARNINGS

#endif // IS_UTF8_H

#include <climits>
#include <initializer_list>

// Useful for debugging purposes
namespace is_utf8_internals {
namespace {

template <typename T> std::string toBinaryString(T b) {
  std::string binary = "";
  T mask = T(1) << (sizeof(T) * CHAR_BIT - 1);
  while (mask > 0) {
    binary += ((b & mask) == 0) ? '0' : '1';
    mask >>= 1;
  }
  return binary;
}
} // namespace
} // namespace is_utf8_internals

// Implementations
// The best choice should always come first!

#ifndef IS_UTF8_ARM64_H
#define IS_UTF8_ARM64_H

#ifdef IS_UTF8_FALLBACK_H
#error "arm64.h must be included before fallback.h"
#endif

#ifndef IS_UTF8_IMPLEMENTATION_ARM64
#define IS_UTF8_IMPLEMENTATION_ARM64 (IS_UTF8_IS_ARM64)
#endif

#if IS_UTF8_IMPLEMENTATION_ARM64 &&IS_UTF8_IS_ARM64
#define IS_UTF8_CAN_ALWAYS_RUN_ARM64 1
#else
#define IS_UTF8_CAN_ALWAYS_RUN_ARM64 0
#endif


#if IS_UTF8_IMPLEMENTATION_ARM64

namespace is_utf8_internals {
/**
 * Implementation for NEON (ARMv8).
 */
namespace arm64 {} // namespace arm64
} // namespace is_utf8_internals

#ifndef IS_UTF8_ARM64_IMPLEMENTATION_H
#define IS_UTF8_ARM64_IMPLEMENTATION_H

namespace is_utf8_internals {
namespace arm64 {

class implementation final : public is_utf8_internals::implementation {
public:
  is_utf8_really_inline implementation()
      : is_utf8_internals::implementation("arm64", "ARM NEON",
                                          internal::instruction_set::NEON) {}
  is_utf8_warn_unused bool validate_utf8(const char *buf,
                                         size_t len) const noexcept final;
};

} // namespace arm64
} // namespace is_utf8_internals

#endif // IS_UTF8_ARM64_IMPLEMENTATION_H

// redefining IS_UTF8_IMPLEMENTATION to "arm64"
// #define IS_UTF8_IMPLEMENTATION arm64

// Declarations

#ifndef IS_UTF8_ARM64_INTRINSICS_H
#define IS_UTF8_ARM64_INTRINSICS_H

// This should be the correct header whether
// you use visual studio or other compilers.
#include <arm_neon.h>

#endif //  IS_UTF8_ARM64_INTRINSICS_H

#ifndef IS_UTF8_ARM64_BITMANIPULATION_H
#define IS_UTF8_ARM64_BITMANIPULATION_H

namespace is_utf8_internals {
namespace arm64 {
namespace {} // unnamed namespace
} // namespace arm64
} // namespace is_utf8_internals

#endif // IS_UTF8_ARM64_BITMANIPULATION_H

#ifndef IS_UTF8_ARM64_SIMD_H
#define IS_UTF8_ARM64_SIMD_H

#include <type_traits>

namespace is_utf8_internals {
namespace arm64 {
namespace {
namespace simd {

#ifdef IS_UTF8_REGULAR_VISUAL_STUDIO
namespace {
// Start of private section with Visual Studio workaround

/**
 * make_uint8x16_t initializes a SIMD register (uint8x16_t).
 * This is needed because, incredibly, the syntax uint8x16_t x = {1,2,3...}
 * is not recognized under Visual Studio! This is a workaround.
 * Using a std::initializer_list<uint8_t>  as a parameter resulted in
 * inefficient code. With the current approach, if the parameters are
 * compile-time constants,
 * GNU GCC compiles it to ldr, the same as uint8x16_t x = {1,2,3...}.
 * You should not use this function except for compile-time constants:
 * it is not efficient.
 */
is_utf8_really_inline uint8x16_t make_uint8x16_t(
    uint8_t x1, uint8_t x2, uint8_t x3, uint8_t x4, uint8_t x5, uint8_t x6,
    uint8_t x7, uint8_t x8, uint8_t x9, uint8_t x10, uint8_t x11, uint8_t x12,
    uint8_t x13, uint8_t x14, uint8_t x15, uint8_t x16) {
  // Doing a load like so end ups generating worse code.
  // uint8_t array[16] = {x1, x2, x3, x4, x5, x6, x7, x8,
  //                     x9, x10,x11,x12,x13,x14,x15,x16};
  // return vld1q_u8(array);
  uint8x16_t x{};
  // incredibly, Visual Studio does not allow x[0] = x1
  x = vsetq_lane_u8(x1, x, 0);
  x = vsetq_lane_u8(x2, x, 1);
  x = vsetq_lane_u8(x3, x, 2);
  x = vsetq_lane_u8(x4, x, 3);
  x = vsetq_lane_u8(x5, x, 4);
  x = vsetq_lane_u8(x6, x, 5);
  x = vsetq_lane_u8(x7, x, 6);
  x = vsetq_lane_u8(x8, x, 7);
  x = vsetq_lane_u8(x9, x, 8);
  x = vsetq_lane_u8(x10, x, 9);
  x = vsetq_lane_u8(x11, x, 10);
  x = vsetq_lane_u8(x12, x, 11);
  x = vsetq_lane_u8(x13, x, 12);
  x = vsetq_lane_u8(x14, x, 13);
  x = vsetq_lane_u8(x15, x, 14);
  x = vsetq_lane_u8(x16, x, 15);
  return x;
}

// We have to do the same work for make_int8x16_t
is_utf8_really_inline int8x16_t
make_int8x16_t(int8_t x1, int8_t x2, int8_t x3, int8_t x4, int8_t x5, int8_t x6,
               int8_t x7, int8_t x8, int8_t x9, int8_t x10, int8_t x11,
               int8_t x12, int8_t x13, int8_t x14, int8_t x15, int8_t x16) {
  // Doing a load like so end ups generating worse code.
  // int8_t array[16] = {x1, x2, x3, x4, x5, x6, x7, x8,
  //                     x9, x10,x11,x12,x13,x14,x15,x16};
  // return vld1q_s8(array);
  int8x16_t x{};
  // incredibly, Visual Studio does not allow x[0] = x1
  x = vsetq_lane_s8(x1, x, 0);
  x = vsetq_lane_s8(x2, x, 1);
  x = vsetq_lane_s8(x3, x, 2);
  x = vsetq_lane_s8(x4, x, 3);
  x = vsetq_lane_s8(x5, x, 4);
  x = vsetq_lane_s8(x6, x, 5);
  x = vsetq_lane_s8(x7, x, 6);
  x = vsetq_lane_s8(x8, x, 7);
  x = vsetq_lane_s8(x9, x, 8);
  x = vsetq_lane_s8(x10, x, 9);
  x = vsetq_lane_s8(x11, x, 10);
  x = vsetq_lane_s8(x12, x, 11);
  x = vsetq_lane_s8(x13, x, 12);
  x = vsetq_lane_s8(x14, x, 13);
  x = vsetq_lane_s8(x15, x, 14);
  x = vsetq_lane_s8(x16, x, 15);
  return x;
}

is_utf8_really_inline uint8x8_t make_uint8x8_t(uint8_t x1, uint8_t x2,
                                               uint8_t x3, uint8_t x4,
                                               uint8_t x5, uint8_t x6,
                                               uint8_t x7, uint8_t x8) {
  uint8x8_t x{};
  x = vset_lane_u8(x1, x, 0);
  x = vset_lane_u8(x2, x, 1);
  x = vset_lane_u8(x3, x, 2);
  x = vset_lane_u8(x4, x, 3);
  x = vset_lane_u8(x5, x, 4);
  x = vset_lane_u8(x6, x, 5);
  x = vset_lane_u8(x7, x, 6);
  x = vset_lane_u8(x8, x, 7);
  return x;
}

is_utf8_really_inline uint16x8_t make_uint16x8_t(uint16_t x1, uint16_t x2,
                                                 uint16_t x3, uint16_t x4,
                                                 uint16_t x5, uint16_t x6,
                                                 uint16_t x7, uint16_t x8) {
  uint16x8_t x{};
  x = vsetq_lane_u16(x1, x, 0);
  x = vsetq_lane_u16(x2, x, 1);
  x = vsetq_lane_u16(x3, x, 2);
  x = vsetq_lane_u16(x4, x, 3);
  x = vsetq_lane_u16(x5, x, 4);
  x = vsetq_lane_u16(x6, x, 5);
  x = vsetq_lane_u16(x7, x, 6);
  x = vsetq_lane_u16(x8, x, 7);
  ;
  return x;
}

is_utf8_really_inline int16x8_t make_int16x8_t(int16_t x1, int16_t x2,
                                               int16_t x3, int16_t x4,
                                               int16_t x5, int16_t x6,
                                               int16_t x7, int16_t x8) {
  uint16x8_t x{};
  x = vsetq_lane_s16(x1, x, 0);
  x = vsetq_lane_s16(x2, x, 1);
  x = vsetq_lane_s16(x3, x, 2);
  x = vsetq_lane_s16(x4, x, 3);
  x = vsetq_lane_s16(x5, x, 4);
  x = vsetq_lane_s16(x6, x, 5);
  x = vsetq_lane_s16(x7, x, 6);
  x = vsetq_lane_s16(x8, x, 7);
  ;
  return x;
}

// End of private section with Visual Studio workaround
} // namespace
#endif // IS_UTF8_REGULAR_VISUAL_STUDIO

template <typename T> struct simd8;

//
// Base class of simd8<uint8_t> and simd8<bool>, both of which use uint8x16_t
// internally.
//
template <typename T, typename Mask = simd8<bool>> struct base_u8 {
  uint8x16_t value;
  static const int SIZE = sizeof(value);

  // Conversion from/to SIMD register
  is_utf8_really_inline base_u8(const uint8x16_t _value) : value(_value) {}
  is_utf8_really_inline operator const uint8x16_t &() const {
    return this->value;
  }
  is_utf8_really_inline operator uint8x16_t &() { return this->value; }
  is_utf8_really_inline T first() const { return vgetq_lane_u8(*this, 0); }
  is_utf8_really_inline T last() const { return vgetq_lane_u8(*this, 15); }

  // Bit operations
  is_utf8_really_inline simd8<T> operator|(const simd8<T> other) const {
    return vorrq_u8(*this, other);
  }
  is_utf8_really_inline simd8<T> operator&(const simd8<T> other) const {
    return vandq_u8(*this, other);
  }
  is_utf8_really_inline simd8<T> operator^(const simd8<T> other) const {
    return veorq_u8(*this, other);
  }
  is_utf8_really_inline simd8<T> bit_andnot(const simd8<T> other) const {
    return vbicq_u8(*this, other);
  }
  is_utf8_really_inline simd8<T> operator~() const { return *this ^ 0xFFu; }
  is_utf8_really_inline simd8<T> &operator|=(const simd8<T> other) {
    auto this_cast = static_cast<simd8<T> *>(this);
    *this_cast = *this_cast | other;
    return *this_cast;
  }
  is_utf8_really_inline simd8<T> &operator&=(const simd8<T> other) {
    auto this_cast = static_cast<simd8<T> *>(this);
    *this_cast = *this_cast & other;
    return *this_cast;
  }
  is_utf8_really_inline simd8<T> &operator^=(const simd8<T> other) {
    auto this_cast = static_cast<simd8<T> *>(this);
    *this_cast = *this_cast ^ other;
    return *this_cast;
  }

  friend is_utf8_really_inline Mask operator==(const simd8<T> lhs,
                                               const simd8<T> rhs) {
    return vceqq_u8(lhs, rhs);
  }

  template <int N = 1>
  is_utf8_really_inline simd8<T> prev(const simd8<T> prev_chunk) const {
    return vextq_u8(prev_chunk, *this, 16 - N);
  }
};

// SIMD byte mask type (returned by things like eq and gt)
template <> struct simd8<bool> : base_u8<bool> {
  typedef uint16_t bitmask_t;
  typedef uint32_t bitmask2_t;

  static is_utf8_really_inline simd8<bool> splat(bool _value) {
    return vmovq_n_u8(uint8_t(-(!!_value)));
  }

  is_utf8_really_inline simd8(const uint8x16_t _value)
      : base_u8<bool>(_value) {}
  // False constructor
  is_utf8_really_inline simd8() : simd8(vdupq_n_u8(0)) {}
  // Splat constructor
  is_utf8_really_inline simd8(bool _value) : simd8(splat(_value)) {}
  is_utf8_really_inline void store(uint8_t dst[16]) const {
    return vst1q_u8(dst, *this);
  }

  // We return uint32_t instead of uint16_t because that seems to be more
  // efficient for most purposes (cutting it down to uint16_t costs performance
  // in some compilers).
  is_utf8_really_inline uint32_t to_bitmask() const {
#ifdef IS_UTF8_REGULAR_VISUAL_STUDIO
    const uint8x16_t bit_mask =
        make_uint8x16_t(0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01,
                        0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80);
#else
    const uint8x16_t bit_mask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
                                 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
#endif
    auto minput = *this & bit_mask;
    uint8x16_t tmp = vpaddq_u8(minput, minput);
    tmp = vpaddq_u8(tmp, tmp);
    tmp = vpaddq_u8(tmp, tmp);
    return vgetq_lane_u16(vreinterpretq_u16_u8(tmp), 0);
  }

  // Returns 4-bit out of each byte, alternating between the high 4 bits and low
  // bits result it is 64 bit. This method is expected to be faster than none()
  // and is equivalent when the vector register is the result of a comparison,
  // with byte values 0xff and 0x00.
  is_utf8_really_inline uint64_t to_bitmask64() const {
    return vget_lane_u64(
        vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(*this), 4)), 0);
  }

  is_utf8_really_inline bool any() const { return vmaxvq_u8(*this) != 0; }
  is_utf8_really_inline bool none() const { return vmaxvq_u8(*this) == 0; }
  is_utf8_really_inline bool all() const { return vminvq_u8(*this) == 0xFF; }
};

// Unsigned bytes
template <> struct simd8<uint8_t> : base_u8<uint8_t> {
  static is_utf8_really_inline simd8<uint8_t> splat(uint8_t _value) {
    return vmovq_n_u8(_value);
  }
  static is_utf8_really_inline simd8<uint8_t> zero() { return vdupq_n_u8(0); }
  static is_utf8_really_inline simd8<uint8_t> load(const uint8_t *values) {
    return vld1q_u8(values);
  }
  is_utf8_really_inline simd8(const uint8x16_t _value)
      : base_u8<uint8_t>(_value) {}
  // Zero constructor
  is_utf8_really_inline simd8() : simd8(zero()) {}
  // Array constructor
  is_utf8_really_inline simd8(const uint8_t values[16]) : simd8(load(values)) {}
  // Splat constructor
  is_utf8_really_inline simd8(uint8_t _value) : simd8(splat(_value)) {}
  // Member-by-member initialization
#ifdef IS_UTF8_REGULAR_VISUAL_STUDIO
  is_utf8_really_inline
  simd8(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4, uint8_t v5,
        uint8_t v6, uint8_t v7, uint8_t v8, uint8_t v9, uint8_t v10,
        uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15)
      : simd8(make_uint8x16_t(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                              v12, v13, v14, v15)) {}
#else
  is_utf8_really_inline
  simd8(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4, uint8_t v5,
        uint8_t v6, uint8_t v7, uint8_t v8, uint8_t v9, uint8_t v10,
        uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15)
      : simd8(uint8x16_t{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
                         v13, v14, v15}) {}
#endif

  // Repeat 16 values as many times as necessary (usually for lookup tables)
  is_utf8_really_inline static simd8<uint8_t>
  repeat_16(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4,
            uint8_t v5, uint8_t v6, uint8_t v7, uint8_t v8, uint8_t v9,
            uint8_t v10, uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14,
            uint8_t v15) {
    return simd8<uint8_t>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
                          v13, v14, v15);
  }

  // Store to array
  is_utf8_really_inline void store(uint8_t dst[16]) const {
    return vst1q_u8(dst, *this);
  }

  // Saturated math
  is_utf8_really_inline simd8<uint8_t>
  saturating_add(const simd8<uint8_t> other) const {
    return vqaddq_u8(*this, other);
  }
  is_utf8_really_inline simd8<uint8_t>
  saturating_sub(const simd8<uint8_t> other) const {
    return vqsubq_u8(*this, other);
  }

  // Addition/subtraction are the same for signed and unsigned
  is_utf8_really_inline simd8<uint8_t>
  operator+(const simd8<uint8_t> other) const {
    return vaddq_u8(*this, other);
  }
  is_utf8_really_inline simd8<uint8_t>
  operator-(const simd8<uint8_t> other) const {
    return vsubq_u8(*this, other);
  }
  is_utf8_really_inline simd8<uint8_t> &operator+=(const simd8<uint8_t> other) {
    *this = *this + other;
    return *this;
  }
  is_utf8_really_inline simd8<uint8_t> &operator-=(const simd8<uint8_t> other) {
    *this = *this - other;
    return *this;
  }

  // Order-specific operations
  is_utf8_really_inline uint8_t max_val() const { return vmaxvq_u8(*this); }
  is_utf8_really_inline uint8_t min_val() const { return vminvq_u8(*this); }
  is_utf8_really_inline simd8<uint8_t>
  max_val(const simd8<uint8_t> other) const {
    return vmaxq_u8(*this, other);
  }
  is_utf8_really_inline simd8<uint8_t>
  min_val(const simd8<uint8_t> other) const {
    return vminq_u8(*this, other);
  }
  is_utf8_really_inline simd8<bool>
  operator<=(const simd8<uint8_t> other) const {
    return vcleq_u8(*this, other);
  }
  is_utf8_really_inline simd8<bool>
  operator>=(const simd8<uint8_t> other) const {
    return vcgeq_u8(*this, other);
  }
  is_utf8_really_inline simd8<bool>
  operator<(const simd8<uint8_t> other) const {
    return vcltq_u8(*this, other);
  }
  is_utf8_really_inline simd8<bool>
  operator>(const simd8<uint8_t> other) const {
    return vcgtq_u8(*this, other);
  }
  // Same as >, but instead of guaranteeing all 1's == true, false = 0 and true
  // = nonzero. For ARM, returns all 1's.
  is_utf8_really_inline simd8<uint8_t>
  gt_bits(const simd8<uint8_t> other) const {
    return simd8<uint8_t>(*this > other);
  }
  // Same as <, but instead of guaranteeing all 1's == true, false = 0 and true
  // = nonzero. For ARM, returns all 1's.
  is_utf8_really_inline simd8<uint8_t>
  lt_bits(const simd8<uint8_t> other) const {
    return simd8<uint8_t>(*this < other);
  }

  // Bit-specific operations
  is_utf8_really_inline simd8<bool> any_bits_set(simd8<uint8_t> bits) const {
    return vtstq_u8(*this, bits);
  }
  is_utf8_really_inline bool is_ascii() const {
    return this->max_val() < 0b10000000u;
  }

  is_utf8_really_inline bool any_bits_set_anywhere() const {
    return this->max_val() != 0;
  }
  is_utf8_really_inline bool any_bits_set_anywhere(simd8<uint8_t> bits) const {
    return (*this & bits).any_bits_set_anywhere();
  }
  template <int N> is_utf8_really_inline simd8<uint8_t> shr() const {
    return vshrq_n_u8(*this, N);
  }
  template <int N> is_utf8_really_inline simd8<uint8_t> shl() const {
    return vshlq_n_u8(*this, N);
  }

  // Perform a lookup assuming the value is between 0 and 16 (undefined behavior
  // for out of range values)
  template <typename L>
  is_utf8_really_inline simd8<L> lookup_16(simd8<L> lookup_table) const {
    return lookup_table.apply_lookup_16_to(*this);
  }

  template <typename L>
  is_utf8_really_inline simd8<L>
  lookup_16(L replace0, L replace1, L replace2, L replace3, L replace4,
            L replace5, L replace6, L replace7, L replace8, L replace9,
            L replace10, L replace11, L replace12, L replace13, L replace14,
            L replace15) const {
    return lookup_16(simd8<L>::repeat_16(
        replace0, replace1, replace2, replace3, replace4, replace5, replace6,
        replace7, replace8, replace9, replace10, replace11, replace12,
        replace13, replace14, replace15));
  }

  template <typename T>
  is_utf8_really_inline simd8<uint8_t>
  apply_lookup_16_to(const simd8<T> original) const {
    return vqtbl1q_u8(*this, simd8<uint8_t>(original));
  }
};

// Signed bytes
template <> struct simd8<int8_t> {
  int8x16_t value;

  static is_utf8_really_inline simd8<int8_t> splat(int8_t _value) {
    return vmovq_n_s8(_value);
  }
  static is_utf8_really_inline simd8<int8_t> zero() { return vdupq_n_s8(0); }
  static is_utf8_really_inline simd8<int8_t> load(const int8_t values[16]) {
    return vld1q_s8(values);
  }
  template <endianness big_endian>
  is_utf8_really_inline void store_ascii_as_utf16(char16_t *p) const {
    uint16x8_t first = vmovl_u8(vget_low_u8(vreinterpretq_u8_s8(this->value)));
    uint16x8_t second = vmovl_high_u8(vreinterpretq_u8_s8(this->value));
    if (big_endian) {
#ifdef IS_UTF8_REGULAR_VISUAL_STUDIO
      const uint8x16_t swap =
          make_uint8x16_t(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
#else
      const uint8x16_t swap = {1, 0, 3,  2,  5,  4,  7,  6,
                               9, 8, 11, 10, 13, 12, 15, 14};
#endif
      first =
          vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(first), swap));
      second =
          vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(second), swap));
    }
    vst1q_u16(reinterpret_cast<uint16_t *>(p), first);
    vst1q_u16(reinterpret_cast<uint16_t *>(p + 8), second);
  }
  is_utf8_really_inline void store_ascii_as_utf32(char32_t *p) const {
    vst1q_u32(reinterpret_cast<uint32_t *>(p),
              vmovl_u16(vget_low_u16(
                  vmovl_u8(vget_low_u8(vreinterpretq_u8_s8(this->value))))));
    vst1q_u32(reinterpret_cast<uint32_t *>(p + 4),
              vmovl_high_u16(
                  vmovl_u8(vget_low_u8(vreinterpretq_u8_s8(this->value)))));
    vst1q_u32(reinterpret_cast<uint32_t *>(p + 8),
              vmovl_u16(vget_low_u16(
                  vmovl_high_u8(vreinterpretq_u8_s8(this->value)))));
    vst1q_u32(reinterpret_cast<uint32_t *>(p + 12),
              vmovl_high_u16(vmovl_high_u8(vreinterpretq_u8_s8(this->value))));
  }
  // Conversion from/to SIMD register
  is_utf8_really_inline simd8(const int8x16_t _value) : value{_value} {}
  is_utf8_really_inline operator const int8x16_t &() const {
    return this->value;
  }
  is_utf8_really_inline operator const uint8x16_t() const {
    return vreinterpretq_u8_s8(this->value);
  }
  is_utf8_really_inline operator int8x16_t &() { return this->value; }

  // Zero constructor
  is_utf8_really_inline simd8() : simd8(zero()) {}
  // Splat constructor
  is_utf8_really_inline simd8(int8_t _value) : simd8(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd8(const int8_t *values) : simd8(load(values)) {}
  // Member-by-member initialization
#ifdef IS_UTF8_REGULAR_VISUAL_STUDIO
  is_utf8_really_inline simd8(int8_t v0, int8_t v1, int8_t v2, int8_t v3,
                              int8_t v4, int8_t v5, int8_t v6, int8_t v7,
                              int8_t v8, int8_t v9, int8_t v10, int8_t v11,
                              int8_t v12, int8_t v13, int8_t v14, int8_t v15)
      : simd8(make_int8x16_t(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                             v12, v13, v14, v15)) {}
#else
  is_utf8_really_inline simd8(int8_t v0, int8_t v1, int8_t v2, int8_t v3,
                              int8_t v4, int8_t v5, int8_t v6, int8_t v7,
                              int8_t v8, int8_t v9, int8_t v10, int8_t v11,
                              int8_t v12, int8_t v13, int8_t v14, int8_t v15)
      : simd8(int8x16_t{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
                        v13, v14, v15}) {}
#endif
  // Repeat 16 values as many times as necessary (usually for lookup tables)
  is_utf8_really_inline static simd8<int8_t>
  repeat_16(int8_t v0, int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5,
            int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11,
            int8_t v12, int8_t v13, int8_t v14, int8_t v15) {
    return simd8<int8_t>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
                         v13, v14, v15);
  }

  // Store to array
  is_utf8_really_inline void store(int8_t dst[16]) const {
    return vst1q_s8(dst, value);
  }
  // Explicit conversion to/from unsigned
  //
  // Under Visual Studio/ARM64 uint8x16_t and int8x16_t are apparently the same
  // type. In theory, we could check this occurrence with std::same_as and
  // std::enabled_if but it is C++14 and relatively ugly and hard to read.
#ifndef IS_UTF8_REGULAR_VISUAL_STUDIO
  is_utf8_really_inline explicit simd8(const uint8x16_t other)
      : simd8(vreinterpretq_s8_u8(other)) {}
#endif
  is_utf8_really_inline operator simd8<uint8_t>() const {
    return vreinterpretq_u8_s8(this->value);
  }

  is_utf8_really_inline simd8<int8_t>
  operator|(const simd8<int8_t> other) const {
    return vorrq_s8(value, other.value);
  }
  is_utf8_really_inline simd8<int8_t>
  operator&(const simd8<int8_t> other) const {
    return vandq_s8(value, other.value);
  }
  is_utf8_really_inline simd8<int8_t>
  operator^(const simd8<int8_t> other) const {
    return veorq_s8(value, other.value);
  }
  is_utf8_really_inline simd8<int8_t>
  bit_andnot(const simd8<int8_t> other) const {
    return vbicq_s8(value, other.value);
  }

  // Math
  is_utf8_really_inline simd8<int8_t>
  operator+(const simd8<int8_t> other) const {
    return vaddq_s8(value, other.value);
  }
  is_utf8_really_inline simd8<int8_t>
  operator-(const simd8<int8_t> other) const {
    return vsubq_s8(value, other.value);
  }
  is_utf8_really_inline simd8<int8_t> &operator+=(const simd8<int8_t> other) {
    *this = *this + other;
    return *this;
  }
  is_utf8_really_inline simd8<int8_t> &operator-=(const simd8<int8_t> other) {
    *this = *this - other;
    return *this;
  }

  is_utf8_really_inline int8_t max_val() const { return vmaxvq_s8(value); }
  is_utf8_really_inline int8_t min_val() const { return vminvq_s8(value); }
  is_utf8_really_inline bool is_ascii() const { return this->min_val() >= 0; }

  // Order-sensitive comparisons
  is_utf8_really_inline simd8<int8_t> max_val(const simd8<int8_t> other) const {
    return vmaxq_s8(value, other.value);
  }
  is_utf8_really_inline simd8<int8_t> min_val(const simd8<int8_t> other) const {
    return vminq_s8(value, other.value);
  }
  is_utf8_really_inline simd8<bool> operator>(const simd8<int8_t> other) const {
    return vcgtq_s8(value, other.value);
  }
  is_utf8_really_inline simd8<bool> operator<(const simd8<int8_t> other) const {
    return vcltq_s8(value, other.value);
  }
  is_utf8_really_inline simd8<bool>
  operator==(const simd8<int8_t> other) const {
    return vceqq_s8(value, other.value);
  }

  template <int N = 1>
  is_utf8_really_inline simd8<int8_t>
  prev(const simd8<int8_t> prev_chunk) const {
    return vextq_s8(prev_chunk, *this, 16 - N);
  }

  // Perform a lookup assuming no value is larger than 16
  template <typename L>
  is_utf8_really_inline simd8<L> lookup_16(simd8<L> lookup_table) const {
    return lookup_table.apply_lookup_16_to(*this);
  }
  template <typename L>
  is_utf8_really_inline simd8<L>
  lookup_16(L replace0, L replace1, L replace2, L replace3, L replace4,
            L replace5, L replace6, L replace7, L replace8, L replace9,
            L replace10, L replace11, L replace12, L replace13, L replace14,
            L replace15) const {
    return lookup_16(simd8<L>::repeat_16(
        replace0, replace1, replace2, replace3, replace4, replace5, replace6,
        replace7, replace8, replace9, replace10, replace11, replace12,
        replace13, replace14, replace15));
  }

  template <typename T>
  is_utf8_really_inline simd8<int8_t>
  apply_lookup_16_to(const simd8<T> original) {
    return vqtbl1q_s8(*this, simd8<uint8_t>(original));
  }
};

template <typename T> struct simd8x64 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd8<T>);
  static_assert(NUM_CHUNKS == 4,
                "ARM kernel should use four registers per 64-byte block.");
  simd8<T> chunks[NUM_CHUNKS];

  simd8x64(const simd8x64<T> &o) = delete; // no copy allowed
  simd8x64<T> &
  operator=(const simd8<T> other) = delete; // no assignment allowed
  simd8x64() = delete;                      // no default constructor allowed

  is_utf8_really_inline simd8x64(const simd8<T> chunk0, const simd8<T> chunk1,
                                 const simd8<T> chunk2, const simd8<T> chunk3)
      : chunks{chunk0, chunk1, chunk2, chunk3} {}
  is_utf8_really_inline simd8x64(const T *ptr)
      : chunks{simd8<T>::load(ptr),
               simd8<T>::load(ptr + sizeof(simd8<T>) / sizeof(T)),
               simd8<T>::load(ptr + 2 * sizeof(simd8<T>) / sizeof(T)),
               simd8<T>::load(ptr + 3 * sizeof(simd8<T>) / sizeof(T))} {}

  is_utf8_really_inline void store(T *ptr) const {
    this->chunks[0].store(ptr + sizeof(simd8<T>) * 0 / sizeof(T));
    this->chunks[1].store(ptr + sizeof(simd8<T>) * 1 / sizeof(T));
    this->chunks[2].store(ptr + sizeof(simd8<T>) * 2 / sizeof(T));
    this->chunks[3].store(ptr + sizeof(simd8<T>) * 3 / sizeof(T));
  }

  is_utf8_really_inline simd8x64<T> &operator|=(const simd8x64<T> &other) {
    this->chunks[0] |= other.chunks[0];
    this->chunks[1] |= other.chunks[1];
    this->chunks[2] |= other.chunks[2];
    this->chunks[3] |= other.chunks[3];
    return *this;
  }

  is_utf8_really_inline simd8<T> reduce_or() const {
    return (this->chunks[0] | this->chunks[1]) |
           (this->chunks[2] | this->chunks[3]);
  }

  is_utf8_really_inline bool is_ascii() const { return reduce_or().is_ascii(); }

  template <endianness endian>
  is_utf8_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    this->chunks[0].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 0);
    this->chunks[1].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 1);
    this->chunks[2].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 2);
    this->chunks[3].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 3);
  }

  is_utf8_really_inline void store_ascii_as_utf32(char32_t *ptr) const {
    this->chunks[0].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 0);
    this->chunks[1].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 1);
    this->chunks[2].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 2);
    this->chunks[3].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 3);
  }

  is_utf8_really_inline uint64_t to_bitmask() const {
#ifdef IS_UTF8_REGULAR_VISUAL_STUDIO
    const uint8x16_t bit_mask =
        make_uint8x16_t(0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01,
                        0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80);
#else
    const uint8x16_t bit_mask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
                                 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
#endif
    // Add each of the elements next to each other, successively, to stuff each
    // 8 byte mask into one.
    uint8x16_t sum0 =
        vpaddq_u8(vandq_u8(uint8x16_t(this->chunks[0]), bit_mask),
                  vandq_u8(uint8x16_t(this->chunks[1]), bit_mask));
    uint8x16_t sum1 =
        vpaddq_u8(vandq_u8(uint8x16_t(this->chunks[2]), bit_mask),
                  vandq_u8(uint8x16_t(this->chunks[3]), bit_mask));
    sum0 = vpaddq_u8(sum0, sum1);
    sum0 = vpaddq_u8(sum0, sum0);
    return vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0);
  }

  is_utf8_really_inline uint64_t eq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] == mask, this->chunks[1] == mask,
                          this->chunks[2] == mask, this->chunks[3] == mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t lteq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] <= mask, this->chunks[1] <= mask,
                          this->chunks[2] <= mask, this->chunks[3] <= mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t in_range(const T low, const T high) const {
    const simd8<T> mask_low = simd8<T>::splat(low);
    const simd8<T> mask_high = simd8<T>::splat(high);

    return simd8x64<bool>(
               (this->chunks[0] <= mask_high) & (this->chunks[0] >= mask_low),
               (this->chunks[1] <= mask_high) & (this->chunks[1] >= mask_low),
               (this->chunks[2] <= mask_high) & (this->chunks[2] >= mask_low),
               (this->chunks[3] <= mask_high) & (this->chunks[3] >= mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t not_in_range(const T low, const T high) const {
    const simd8<T> mask_low = simd8<T>::splat(low);
    const simd8<T> mask_high = simd8<T>::splat(high);
    return simd8x64<bool>(
               (this->chunks[0] > mask_high) | (this->chunks[0] < mask_low),
               (this->chunks[1] > mask_high) | (this->chunks[1] < mask_low),
               (this->chunks[2] > mask_high) | (this->chunks[2] < mask_low),
               (this->chunks[3] > mask_high) | (this->chunks[3] < mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t lt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] < mask, this->chunks[1] < mask,
                          this->chunks[2] < mask, this->chunks[3] < mask)
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t gt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] > mask, this->chunks[1] > mask,
                          this->chunks[2] > mask, this->chunks[3] > mask)
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t gteq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] >= mask, this->chunks[1] >= mask,
                          this->chunks[2] >= mask, this->chunks[3] >= mask)
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t gteq_unsigned(const uint8_t m) const {
    const simd8<uint8_t> mask = simd8<uint8_t>::splat(m);
    return simd8x64<bool>(simd8<uint8_t>(uint8x16_t(this->chunks[0])) >= mask,
                          simd8<uint8_t>(uint8x16_t(this->chunks[1])) >= mask,
                          simd8<uint8_t>(uint8x16_t(this->chunks[2])) >= mask,
                          simd8<uint8_t>(uint8x16_t(this->chunks[3])) >= mask)
        .to_bitmask();
  }
}; // struct simd8x64<T>

template <typename T> struct simd16;

template <typename T, typename Mask = simd16<bool>> struct base_u16 {
  uint16x8_t value;
  static const int SIZE = sizeof(value);

  // Conversion from/to SIMD register
  is_utf8_really_inline base_u16() = default;
  is_utf8_really_inline base_u16(const uint16x8_t _value) : value(_value) {}
  is_utf8_really_inline operator const uint16x8_t &() const {
    return this->value;
  }
  is_utf8_really_inline operator uint16x8_t &() { return this->value; }
  // Bit operations
  is_utf8_really_inline simd16<T> operator|(const simd16<T> other) const {
    return vorrq_u16(*this, other);
  }
  is_utf8_really_inline simd16<T> operator&(const simd16<T> other) const {
    return vandq_u16(*this, other);
  }
  is_utf8_really_inline simd16<T> operator^(const simd16<T> other) const {
    return veorq_u16(*this, other);
  }
  is_utf8_really_inline simd16<T> bit_andnot(const simd16<T> other) const {
    return vbicq_u16(*this, other);
  }
  is_utf8_really_inline simd16<T> operator~() const { return *this ^ 0xFFu; }
  is_utf8_really_inline simd16<T> &operator|=(const simd16<T> other) {
    auto this_cast = static_cast<simd16<T> *>(this);
    *this_cast = *this_cast | other;
    return *this_cast;
  }
  is_utf8_really_inline simd16<T> &operator&=(const simd16<T> other) {
    auto this_cast = static_cast<simd16<T> *>(this);
    *this_cast = *this_cast & other;
    return *this_cast;
  }
  is_utf8_really_inline simd16<T> &operator^=(const simd16<T> other) {
    auto this_cast = static_cast<simd16<T> *>(this);
    *this_cast = *this_cast ^ other;
    return *this_cast;
  }

  is_utf8_really_inline Mask operator==(const simd16<T> other) const {
    return vceqq_u16(*this, other);
  }

  template <int N = 1>
  is_utf8_really_inline simd16<T> prev(const simd16<T> prev_chunk) const {
    return vextq_u18(prev_chunk, *this, 8 - N);
  }
};

template <typename T, typename Mask = simd16<bool>>
struct base16 : base_u16<T> {
  typedef uint16_t bitmask_t;
  typedef uint32_t bitmask2_t;

  is_utf8_really_inline base16() : base_u16<T>() {}
  is_utf8_really_inline base16(const uint16x8_t _value) : base_u16<T>(_value) {}
  template <typename Pointer>
  is_utf8_really_inline base16(const Pointer *ptr) : base16(vld1q_u16(ptr)) {}

  is_utf8_really_inline Mask operator==(const simd16<T> other) const {
    return vceqq_u16(*this, other);
  }

  static const int SIZE = sizeof(base_u16<T>::value);

  template <int N = 1>
  is_utf8_really_inline simd16<T> prev(const simd16<T> prev_chunk) const {
    return vextq_u18(prev_chunk, *this, 8 - N);
  }
};

// SIMD byte mask type (returned by things like eq and gt)
template <> struct simd16<bool> : base16<bool> {
  static is_utf8_really_inline simd16<bool> splat(bool _value) {
    return vmovq_n_u16(uint16_t(-(!!_value)));
  }

  is_utf8_really_inline simd16<bool>() : base16() {}
  is_utf8_really_inline simd16<bool>(const uint16x8_t _value)
      : base16<bool>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd16<bool>(bool _value)
      : base16<bool>(splat(_value)) {}
};

template <typename T> struct base16_numeric : base16<T> {
  static is_utf8_really_inline simd16<T> splat(T _value) {
    return vmovq_n_u16(_value);
  }
  static is_utf8_really_inline simd16<T> zero() { return vdupq_n_u16(0); }
  static is_utf8_really_inline simd16<T> load(const T values[8]) {
    return vld1q_u16(reinterpret_cast<const uint16_t *>(values));
  }

  is_utf8_really_inline base16_numeric() : base16<T>() {}
  is_utf8_really_inline base16_numeric(const uint16x8_t _value)
      : base16<T>(_value) {}

  // Store to array
  is_utf8_really_inline void store(T dst[8]) const {
    return vst1q_u16(dst, *this);
  }

  // Override to distinguish from bool version
  is_utf8_really_inline simd16<T> operator~() const { return *this ^ 0xFFu; }

  // Addition/subtraction are the same for signed and unsigned
  is_utf8_really_inline simd16<T> operator+(const simd16<T> other) const {
    return vaddq_u8(*this, other);
  }
  is_utf8_really_inline simd16<T> operator-(const simd16<T> other) const {
    return vsubq_u8(*this, other);
  }
  is_utf8_really_inline simd16<T> &operator+=(const simd16<T> other) {
    *this = *this + other;
    return *static_cast<simd16<T> *>(this);
  }
  is_utf8_really_inline simd16<T> &operator-=(const simd16<T> other) {
    *this = *this - other;
    return *static_cast<simd16<T> *>(this);
  }
};

// Signed words
template <> struct simd16<int16_t> : base16_numeric<int16_t> {
  is_utf8_really_inline simd16() : base16_numeric<int16_t>() {}
#ifndef IS_UTF8_REGULAR_VISUAL_STUDIO
  is_utf8_really_inline simd16(const uint16x8_t _value)
      : base16_numeric<int16_t>(_value) {}
#endif
  is_utf8_really_inline simd16(const int16x8_t _value)
      : base16_numeric<int16_t>(vreinterpretq_u16_s16(_value)) {}

  // Splat constructor
  is_utf8_really_inline simd16(int16_t _value) : simd16(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd16(const int16_t *values) : simd16(load(values)) {}
  is_utf8_really_inline simd16(const char16_t *values)
      : simd16(load(reinterpret_cast<const int16_t *>(values))) {}
  is_utf8_really_inline operator simd16<uint16_t>() const;
  is_utf8_really_inline operator const uint16x8_t &() const {
    return this->value;
  }
  is_utf8_really_inline operator const int16x8_t() const {
    return vreinterpretq_s16_u16(this->value);
  }

  is_utf8_really_inline int16_t max_val() const {
    return vmaxvq_s16(vreinterpretq_s16_u16(this->value));
  }
  is_utf8_really_inline int16_t min_val() const {
    return vminvq_s16(vreinterpretq_s16_u16(this->value));
  }
  // Order-sensitive comparisons
  is_utf8_really_inline simd16<int16_t>
  max_val(const simd16<int16_t> other) const {
    return vmaxq_s16(vreinterpretq_s16_u16(this->value),
                     vreinterpretq_s16_u16(other.value));
  }
  is_utf8_really_inline simd16<int16_t>
  min_val(const simd16<int16_t> other) const {
    return vmaxq_s16(vreinterpretq_s16_u16(this->value),
                     vreinterpretq_s16_u16(other.value));
  }
  is_utf8_really_inline simd16<bool>
  operator>(const simd16<int16_t> other) const {
    return vcgtq_s16(vreinterpretq_s16_u16(this->value),
                     vreinterpretq_s16_u16(other.value));
  }
  is_utf8_really_inline simd16<bool>
  operator<(const simd16<int16_t> other) const {
    return vcltq_s16(vreinterpretq_s16_u16(this->value),
                     vreinterpretq_s16_u16(other.value));
  }
};

// Unsigned words
template <> struct simd16<uint16_t> : base16_numeric<uint16_t> {
  is_utf8_really_inline simd16() : base16_numeric<uint16_t>() {}
  is_utf8_really_inline simd16(const uint16x8_t _value)
      : base16_numeric<uint16_t>(_value) {}

  // Splat constructor
  is_utf8_really_inline simd16(uint16_t _value) : simd16(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd16(const uint16_t *values) : simd16(load(values)) {}
  is_utf8_really_inline simd16(const char16_t *values)
      : simd16(load(reinterpret_cast<const uint16_t *>(values))) {}

  is_utf8_really_inline int16_t max_val() const { return vmaxvq_u16(*this); }
  is_utf8_really_inline int16_t min_val() const { return vminvq_u16(*this); }
  // Saturated math
  is_utf8_really_inline simd16<uint16_t>
  saturating_add(const simd16<uint16_t> other) const {
    return vqaddq_u16(*this, other);
  }
  is_utf8_really_inline simd16<uint16_t>
  saturating_sub(const simd16<uint16_t> other) const {
    return vqsubq_u16(*this, other);
  }

  // Order-specific operations
  is_utf8_really_inline simd16<uint16_t>
  max_val(const simd16<uint16_t> other) const {
    return vmaxq_u16(*this, other);
  }
  is_utf8_really_inline simd16<uint16_t>
  min_val(const simd16<uint16_t> other) const {
    return vminq_u16(*this, other);
  }
  // Same as >, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd16<uint16_t>
  gt_bits(const simd16<uint16_t> other) const {
    return this->saturating_sub(other);
  }
  // Same as <, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd16<uint16_t>
  lt_bits(const simd16<uint16_t> other) const {
    return other.saturating_sub(*this);
  }
  is_utf8_really_inline simd16<bool>
  operator<=(const simd16<uint16_t> other) const {
    return vcleq_u16(*this, other);
  }
  is_utf8_really_inline simd16<bool>
  operator>=(const simd16<uint16_t> other) const {
    return vcgeq_u16(*this, other);
  }
  is_utf8_really_inline simd16<bool>
  operator>(const simd16<uint16_t> other) const {
    return vcgtq_u16(*this, other);
  }
  is_utf8_really_inline simd16<bool>
  operator<(const simd16<uint16_t> other) const {
    return vcltq_u16(*this, other);
  }

  // Bit-specific operations
  is_utf8_really_inline simd16<bool> bits_not_set() const {
    return *this == uint16_t(0);
  }
  template <int N> is_utf8_really_inline simd16<uint16_t> shr() const {
    return simd16<uint16_t>(vshrq_n_u16(*this, N));
  }
  template <int N> is_utf8_really_inline simd16<uint16_t> shl() const {
    return simd16<uint16_t>(vshlq_n_u16(*this, N));
  }

  // logical operations
  is_utf8_really_inline simd16<uint16_t>
  operator|(const simd16<uint16_t> other) const {
    return vorrq_u16(*this, other);
  }
  is_utf8_really_inline simd16<uint16_t>
  operator&(const simd16<uint16_t> other) const {
    return vandq_u16(*this, other);
  }
  is_utf8_really_inline simd16<uint16_t>
  operator^(const simd16<uint16_t> other) const {
    return veorq_u16(*this, other);
  }

  // Pack with the unsigned saturation  two uint16_t words into single uint8_t
  // vector
  static is_utf8_really_inline simd8<uint8_t> pack(const simd16<uint16_t> &v0,
                                                   const simd16<uint16_t> &v1) {
    return vqmovn_high_u16(vqmovn_u16(v0), v1);
  }

  // Change the endianness
  is_utf8_really_inline simd16<uint16_t> swap_bytes() const {
#ifdef IS_UTF8_REGULAR_VISUAL_STUDIO
    const uint8x16_t swap =
        make_uint8x16_t(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
#else
    const uint8x16_t swap = {1, 0, 3,  2,  5,  4,  7,  6,
                             9, 8, 11, 10, 13, 12, 15, 14};
#endif
    return vreinterpretq_u16_u8(vqtbl1q_u8(vreinterpretq_u8_u16(*this), swap));
  }
};
is_utf8_really_inline simd16<int16_t>::operator simd16<uint16_t>() const {
  return this->value;
}

template <typename T> struct simd16x32 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd16<T>);
  static_assert(NUM_CHUNKS == 4,
                "ARM kernel should use four registers per 64-byte block.");
  simd16<T> chunks[NUM_CHUNKS];

  simd16x32(const simd16x32<T> &o) = delete; // no copy allowed
  simd16x32<T> &
  operator=(const simd16<T> other) = delete; // no assignment allowed
  simd16x32() = delete;                      // no default constructor allowed

  is_utf8_really_inline
  simd16x32(const simd16<T> chunk0, const simd16<T> chunk1,
            const simd16<T> chunk2, const simd16<T> chunk3)
      : chunks{chunk0, chunk1, chunk2, chunk3} {}
  is_utf8_really_inline simd16x32(const T *ptr)
      : chunks{simd16<T>::load(ptr),
               simd16<T>::load(ptr + sizeof(simd16<T>) / sizeof(T)),
               simd16<T>::load(ptr + 2 * sizeof(simd16<T>) / sizeof(T)),
               simd16<T>::load(ptr + 3 * sizeof(simd16<T>) / sizeof(T))} {}

  is_utf8_really_inline void store(T *ptr) const {
    this->chunks[0].store(ptr + sizeof(simd16<T>) * 0 / sizeof(T));
    this->chunks[1].store(ptr + sizeof(simd16<T>) * 1 / sizeof(T));
    this->chunks[2].store(ptr + sizeof(simd16<T>) * 2 / sizeof(T));
    this->chunks[3].store(ptr + sizeof(simd16<T>) * 3 / sizeof(T));
  }

  is_utf8_really_inline simd16<T> reduce_or() const {
    return (this->chunks[0] | this->chunks[1]) |
           (this->chunks[2] | this->chunks[3]);
  }

  is_utf8_really_inline bool is_ascii() const { return reduce_or().is_ascii(); }

  is_utf8_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    this->chunks[0].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 0);
    this->chunks[1].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 1);
    this->chunks[2].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 2);
    this->chunks[3].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 3);
  }

  is_utf8_really_inline uint64_t to_bitmask() const {
#ifdef IS_UTF8_REGULAR_VISUAL_STUDIO
    const uint8x16_t bit_mask =
        make_uint8x16_t(0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x01,
                        0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80);
#else
    const uint8x16_t bit_mask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
                                 0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
#endif
    // Add each of the elements next to each other, successively, to stuff each
    // 8 byte mask into one.
    uint8x16_t sum0 = vpaddq_u8(
        vreinterpretq_u8_u16(this->chunks[0] & vreinterpretq_u16_u8(bit_mask)),
        vreinterpretq_u8_u16(this->chunks[1] & vreinterpretq_u16_u8(bit_mask)));
    uint8x16_t sum1 = vpaddq_u8(
        vreinterpretq_u8_u16(this->chunks[2] & vreinterpretq_u16_u8(bit_mask)),
        vreinterpretq_u8_u16(this->chunks[3] & vreinterpretq_u16_u8(bit_mask)));
    sum0 = vpaddq_u8(sum0, sum1);
    sum0 = vpaddq_u8(sum0, sum0);
    return vgetq_lane_u64(vreinterpretq_u64_u8(sum0), 0);
  }

  is_utf8_really_inline void swap_bytes() {
    this->chunks[0] = this->chunks[0].swap_bytes();
    this->chunks[1] = this->chunks[1].swap_bytes();
    this->chunks[2] = this->chunks[2].swap_bytes();
    this->chunks[3] = this->chunks[3].swap_bytes();
  }

  is_utf8_really_inline uint64_t eq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] == mask, this->chunks[1] == mask,
                           this->chunks[2] == mask, this->chunks[3] == mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t lteq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] <= mask, this->chunks[1] <= mask,
                           this->chunks[2] <= mask, this->chunks[3] <= mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t in_range(const T low, const T high) const {
    const simd16<T> mask_low = simd16<T>::splat(low);
    const simd16<T> mask_high = simd16<T>::splat(high);

    return simd16x32<bool>(
               (this->chunks[0] <= mask_high) & (this->chunks[0] >= mask_low),
               (this->chunks[1] <= mask_high) & (this->chunks[1] >= mask_low),
               (this->chunks[2] <= mask_high) & (this->chunks[2] >= mask_low),
               (this->chunks[3] <= mask_high) & (this->chunks[3] >= mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t not_in_range(const T low, const T high) const {
    const simd16<T> mask_low = simd16<T>::splat(low);
    const simd16<T> mask_high = simd16<T>::splat(high);
    return simd16x32<bool>(
               (this->chunks[0] > mask_high) | (this->chunks[0] < mask_low),
               (this->chunks[1] > mask_high) | (this->chunks[1] < mask_low),
               (this->chunks[2] > mask_high) | (this->chunks[2] < mask_low),
               (this->chunks[3] > mask_high) | (this->chunks[3] < mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t lt(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] < mask, this->chunks[1] < mask,
                           this->chunks[2] < mask, this->chunks[3] < mask)
        .to_bitmask();
  }

}; // struct simd16x32<T>
template <>
is_utf8_really_inline uint64_t simd16x32<uint16_t>::not_in_range(
    const uint16_t low, const uint16_t high) const {
  const simd16<uint16_t> mask_low = simd16<uint16_t>::splat(low);
  const simd16<uint16_t> mask_high = simd16<uint16_t>::splat(high);
  simd16x32<uint16_t> x(simd16<uint16_t>((this->chunks[0] > mask_high) |
                                         (this->chunks[0] < mask_low)),
                        simd16<uint16_t>((this->chunks[1] > mask_high) |
                                         (this->chunks[1] < mask_low)),
                        simd16<uint16_t>((this->chunks[2] > mask_high) |
                                         (this->chunks[2] < mask_low)),
                        simd16<uint16_t>((this->chunks[3] > mask_high) |
                                         (this->chunks[3] < mask_low)));
  return x.to_bitmask();
}

} // namespace simd
} // unnamed namespace
} // namespace arm64
} // namespace is_utf8_internals

#endif // IS_UTF8_ARM64_SIMD_H

#endif // IS_UTF8_IMPLEMENTATION_ARM64

#endif // IS_UTF8_ARM64_H

#ifndef IS_UTF8_ICELAKE_H
#define IS_UTF8_ICELAKE_H

#ifdef __has_include
// How do we detect that a compiler supports vbmi2?
// For sure if the following header is found, we are ok?
#if __has_include(<avx512vbmi2intrin.h>)
#define IS_UTF8_COMPILER_SUPPORTS_VBMI2 1
#endif
#endif

#ifdef _MSC_VER
#if _MSC_VER >= 1920
// Visual Studio 2019 and up support VBMI2 under x64 even if the header
// avx512vbmi2intrin.h is not found.
#define IS_UTF8_COMPILER_SUPPORTS_VBMI2 1
#endif
#endif

// We allow icelake on x64 as long as the compiler is known to support VBMI2.
#ifndef IS_UTF8_IMPLEMENTATION_ICELAKE
#define IS_UTF8_IMPLEMENTATION_ICELAKE                                         \
  ((IS_UTF8_IS_X86_64) && (IS_UTF8_COMPILER_SUPPORTS_VBMI2))
#endif

// To see why  (__BMI__) && (__PCLMUL__) && (__LZCNT__) are not part of this
// next line, see https://github.com/simdutf/simdutf/issues/1247
#define IS_UTF8_CAN_ALWAYS_RUN_ICELAKE                                         \
  ((IS_UTF8_IMPLEMENTATION_ICELAKE) && (IS_UTF8_IS_X86_64) && (__AVX2__) &&    \
   (IS_UTF8_HAS_AVX512F && IS_UTF8_HAS_AVX512DQ && IS_UTF8_HAS_AVX512VL &&     \
    IS_UTF8_HAS_AVX512VBMI2) &&                                                \
   (!IS_UTF8_IS_32BITS))

#if IS_UTF8_IMPLEMENTATION_ICELAKE
#if IS_UTF8_CAN_ALWAYS_RUN_ICELAKE
#define IS_UTF8_TARGET_ICELAKE
#define IS_UTF8_UNTARGET_ICELAKE
#else
#define IS_UTF8_TARGET_ICELAKE                                                 \
  IS_UTF8_TARGET_REGION("avx512f,avx512dq,avx512cd,avx512bw,avx512vbmi,"       \
                        "avx512vbmi2,avx512vl,avx2,bmi,bmi2,pclmul,lzcnt")
#define IS_UTF8_UNTARGET_ICELAKE IS_UTF8_UNTARGET_REGION
#endif

namespace is_utf8_internals {
namespace icelake {} // namespace icelake
} // namespace is_utf8_internals

//
// These two need to be included outside IS_UTF8_TARGET_REGION
//

#ifndef IS_UTF8_ICELAKE_INTRINSICS_H
#define IS_UTF8_ICELAKE_INTRINSICS_H

#ifdef IS_UTF8_VISUAL_STUDIO
// under clang within visual studio, this will include <x86intrin.h>
#include <immintrin.h>
#include <intrin.h> // visual studio or clang
#else

#if IS_UTF8_GCC11ORMORE
// We should not get warnings while including <x86intrin.h> yet we do
// under some versions of GCC.
// If the x86intrin.h header has uninitialized values that are problematic,
// it is a GCC issue, we want to ignore these warnings.
IS_UTF8_DISABLE_GCC_WARNING(-Wuninitialized)
#endif

#include <x86intrin.h> // elsewhere

#if IS_UTF8_GCC11ORMORE
// cancels the suppression of the -Wuninitialized
IS_UTF8_POP_DISABLE_WARNINGS
#endif

#ifndef _tzcnt_u64
#define _tzcnt_u64(x) __tzcnt_u64(x)
#endif // _tzcnt_u64
#endif // IS_UTF8_VISUAL_STUDIO

#ifdef IS_UTF8_CLANG_VISUAL_STUDIO

#include <avx2intrin.h>
#include <avxintrin.h>
#include <bmi2intrin.h>  // for _pext_u64, _pdep_u64
#include <bmiintrin.h>   // for _blsr_u64
#include <immintrin.h>   // for most things (AVX2, AVX512, _popcnt64)
#include <lzcntintrin.h> // for  __lzcnt64
#include <smmintrin.h>
#include <tmmintrin.h>
#include <wmmintrin.h> // for  _mm_clmulepi64_si128
// Important: we need the AVX-512 headers:
#include <avx512bwintrin.h>
#include <avx512cdintrin.h>
#include <avx512dqintrin.h>
#include <avx512fintrin.h>
#include <avx512vbmi2intrin.h>
#include <avx512vbmiintrin.h>
#include <avx512vlbwintrin.h>
#include <avx512vlintrin.h>
// unfortunately, we may not get _blsr_u64, but, thankfully, clang
// has it as a macro.
#ifndef _blsr_u64
// we roll our own
#define _blsr_u64(n) ((n - 1) & n)
#endif //  _blsr_u64
#endif // IS_UTF8_CLANG_VISUAL_STUDIO

#if defined(__GNUC__) && !defined(__clang__)

#if __GNUC__ == 8
#define IS_UTF8_GCC8 1
#elif __GNUC__ == 9
#define IS_UTF8_GCC9 1
#endif //  __GNUC__ == 8 || __GNUC__ == 9

#endif // defined(__GNUC__) && !defined(__clang__)

#if IS_UTF8_GCC8
#pragma GCC push_options
#pragma GCC target("avx512f")
/**
 * GCC 8 fails to provide _mm512_set_epi8. We roll our own.
 */
inline __m512i
_mm512_set_epi8(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4,
                uint8_t a5, uint8_t a6, uint8_t a7, uint8_t a8, uint8_t a9,
                uint8_t a10, uint8_t a11, uint8_t a12, uint8_t a13, uint8_t a14,
                uint8_t a15, uint8_t a16, uint8_t a17, uint8_t a18, uint8_t a19,
                uint8_t a20, uint8_t a21, uint8_t a22, uint8_t a23, uint8_t a24,
                uint8_t a25, uint8_t a26, uint8_t a27, uint8_t a28, uint8_t a29,
                uint8_t a30, uint8_t a31, uint8_t a32, uint8_t a33, uint8_t a34,
                uint8_t a35, uint8_t a36, uint8_t a37, uint8_t a38, uint8_t a39,
                uint8_t a40, uint8_t a41, uint8_t a42, uint8_t a43, uint8_t a44,
                uint8_t a45, uint8_t a46, uint8_t a47, uint8_t a48, uint8_t a49,
                uint8_t a50, uint8_t a51, uint8_t a52, uint8_t a53, uint8_t a54,
                uint8_t a55, uint8_t a56, uint8_t a57, uint8_t a58, uint8_t a59,
                uint8_t a60, uint8_t a61, uint8_t a62, uint8_t a63) {
  return _mm512_set_epi64(
      uint64_t(a7) + (uint64_t(a6) << 8) + (uint64_t(a5) << 16) +
          (uint64_t(a4) << 24) + (uint64_t(a3) << 32) + (uint64_t(a2) << 40) +
          (uint64_t(a1) << 48) + (uint64_t(a0) << 56),
      uint64_t(a15) + (uint64_t(a14) << 8) + (uint64_t(a13) << 16) +
          (uint64_t(a12) << 24) + (uint64_t(a11) << 32) +
          (uint64_t(a10) << 40) + (uint64_t(a9) << 48) + (uint64_t(a8) << 56),
      uint64_t(a23) + (uint64_t(a22) << 8) + (uint64_t(a21) << 16) +
          (uint64_t(a20) << 24) + (uint64_t(a19) << 32) +
          (uint64_t(a18) << 40) + (uint64_t(a17) << 48) + (uint64_t(a16) << 56),
      uint64_t(a31) + (uint64_t(a30) << 8) + (uint64_t(a29) << 16) +
          (uint64_t(a28) << 24) + (uint64_t(a27) << 32) +
          (uint64_t(a26) << 40) + (uint64_t(a25) << 48) + (uint64_t(a24) << 56),
      uint64_t(a39) + (uint64_t(a38) << 8) + (uint64_t(a37) << 16) +
          (uint64_t(a36) << 24) + (uint64_t(a35) << 32) +
          (uint64_t(a34) << 40) + (uint64_t(a33) << 48) + (uint64_t(a32) << 56),
      uint64_t(a47) + (uint64_t(a46) << 8) + (uint64_t(a45) << 16) +
          (uint64_t(a44) << 24) + (uint64_t(a43) << 32) +
          (uint64_t(a42) << 40) + (uint64_t(a41) << 48) + (uint64_t(a40) << 56),
      uint64_t(a55) + (uint64_t(a54) << 8) + (uint64_t(a53) << 16) +
          (uint64_t(a52) << 24) + (uint64_t(a51) << 32) +
          (uint64_t(a50) << 40) + (uint64_t(a49) << 48) + (uint64_t(a48) << 56),
      uint64_t(a63) + (uint64_t(a62) << 8) + (uint64_t(a61) << 16) +
          (uint64_t(a60) << 24) + (uint64_t(a59) << 32) +
          (uint64_t(a58) << 40) + (uint64_t(a57) << 48) +
          (uint64_t(a56) << 56));
}
#pragma GCC pop_options
#endif // IS_UTF8_GCC8

#endif // IS_UTF8_HASWELL_INTRINSICS_H

#ifndef IS_UTF8_ICELAKE_IMPLEMENTATION_H
#define IS_UTF8_ICELAKE_IMPLEMENTATION_H

namespace is_utf8_internals {
namespace icelake {

class implementation final : public is_utf8_internals::implementation {
public:
  is_utf8_really_inline implementation()
      : is_utf8_internals::implementation(
            "icelake",
            "Intel AVX512 (AVX-512BW, AVX-512CD, AVX-512VL, AVX-512VBMI2 "
            "extensions)",
            internal::instruction_set::AVX2 |
                internal::instruction_set::PCLMULQDQ |
                internal::instruction_set::BMI1 |
                internal::instruction_set::BMI2 |
                internal::instruction_set::AVX512BW |
                internal::instruction_set::AVX512CD |
                internal::instruction_set::AVX512VL |
                internal::instruction_set::AVX512VBMI2) {}
  is_utf8_warn_unused bool validate_utf8(const char *buf,
                                         size_t len) const noexcept final;
};

} // namespace icelake
} // namespace is_utf8_internals

#endif // IS_UTF8_ICELAKE_IMPLEMENTATION_H

#if IS_UTF8_GCC11ORMORE // workaround for
                        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
IS_UTF8_POP_DISABLE_WARNINGS
#endif // end of workaround

#endif // IS_UTF8_IMPLEMENTATION_ICELAKE
#endif // IS_UTF8_ICELAKE_H

#ifndef IS_UTF8_HASWELL_H
#define IS_UTF8_HASWELL_H

#ifdef IS_UTF8_WESTMERE_H
#error "haswell.h must be included before westmere.h"
#endif
#ifdef IS_UTF8_FALLBACK_H
#error "haswell.h must be included before fallback.h"
#endif

// Default Haswell to on if this is x86-64. Even if we're not compiled for it,
// it could be selected at runtime.
#ifndef IS_UTF8_IMPLEMENTATION_HASWELL
//
// You do not want to restrict it like so: IS_UTF8_IS_X86_64 && __AVX2__
// because we want to rely on *runtime dispatch*.
//
#if IS_UTF8_CAN_ALWAYS_RUN_ICELAKE
#define IS_UTF8_IMPLEMENTATION_HASWELL 0
#else
#define IS_UTF8_IMPLEMENTATION_HASWELL (IS_UTF8_IS_X86_64)
#endif

#endif
// To see why  (__BMI__) && (__PCLMUL__) && (__LZCNT__) are not part of this
// next line, see https://github.com/simdutf/simdutf/issues/1247
#if ((IS_UTF8_IMPLEMENTATION_HASWELL) && (IS_UTF8_IS_X86_64) && (__AVX2__))
#define IS_UTF8_CAN_ALWAYS_RUN_HASWELL  1
#else
#define IS_UTF8_CAN_ALWAYS_RUN_HASWELL  0
#endif

#if IS_UTF8_IMPLEMENTATION_HASWELL

#define IS_UTF8_TARGET_HASWELL IS_UTF8_TARGET_REGION("avx2,bmi,pclmul,lzcnt")

namespace is_utf8_internals {
/**
 * Implementation for Haswell (Intel AVX2).
 */
namespace haswell {} // namespace haswell
} // namespace is_utf8_internals

//
// These two need to be included outside IS_UTF8_TARGET_REGION
//

#ifndef IS_UTF8_HASWELL_IMPLEMENTATION_H
#define IS_UTF8_HASWELL_IMPLEMENTATION_H

// The constructor may be executed on any host, so we take care not to use
// IS_UTF8_TARGET_REGION
namespace is_utf8_internals {
namespace haswell {

class implementation final : public is_utf8_internals::implementation {
public:
  is_utf8_really_inline implementation()
      : is_utf8_internals::implementation(
            "haswell", "Intel/AMD AVX2",
            internal::instruction_set::AVX2 |
                internal::instruction_set::PCLMULQDQ |
                internal::instruction_set::BMI1 |
                internal::instruction_set::BMI2) {}
  is_utf8_warn_unused bool validate_utf8(const char *buf,
                                         size_t len) const noexcept final;
};

} // namespace haswell
} // namespace is_utf8_internals

#endif // IS_UTF8_HASWELL_IMPLEMENTATION_H

#ifndef IS_UTF8_HASWELL_INTRINSICS_H
#define IS_UTF8_HASWELL_INTRINSICS_H

#ifdef IS_UTF8_VISUAL_STUDIO
// under clang within visual studio, this will include <x86intrin.h>
#include <intrin.h> // visual studio or clang
#else

#if IS_UTF8_GCC11ORMORE
// We should not get warnings while including <x86intrin.h> yet we do
// under some versions of GCC.
// If the x86intrin.h header has uninitialized values that are problematic,
// it is a GCC issue, we want to ignore these warnings.
IS_UTF8_DISABLE_GCC_WARNING(-Wuninitialized)
#endif

#include <x86intrin.h> // elsewhere

#if IS_UTF8_GCC11ORMORE
// cancels the suppression of the -Wuninitialized
IS_UTF8_POP_DISABLE_WARNINGS
#endif

#endif // IS_UTF8_VISUAL_STUDIO

#ifdef IS_UTF8_CLANG_VISUAL_STUDIO

#include <avx2intrin.h>
#include <avxintrin.h>
#include <bmiintrin.h>   // for _blsr_u64
#include <immintrin.h>   // for most things (AVX2, AVX512, _popcnt64)
#include <lzcntintrin.h> // for  __lzcnt64
#include <smmintrin.h>
#include <tmmintrin.h>
#include <wmmintrin.h> // for  _mm_clmulepi64_si128
// unfortunately, we may not get _blsr_u64, but, thankfully, clang
// has it as a macro.
#ifndef _blsr_u64
// we roll our own
#define _blsr_u64(n) ((n - 1) & n)
#endif //  _blsr_u64
#endif // IS_UTF8_CLANG_VISUAL_STUDIO

#endif // IS_UTF8_HASWELL_INTRINSICS_H

//
// The rest need to be inside the region
//

// redefining IS_UTF8_IMPLEMENTATION to "haswell"
// #define IS_UTF8_IMPLEMENTATION haswell
IS_UTF8_TARGET_HASWELL

#if IS_UTF8_GCC11ORMORE // workaround for
                        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
IS_UTF8_DISABLE_GCC_WARNING(-Wmaybe-uninitialized)
#endif // end of workaround

// Declarations

#ifndef IS_UTF8_HASWELL_SIMD_H
#define IS_UTF8_HASWELL_SIMD_H

namespace is_utf8_internals {
namespace haswell {
namespace {
namespace simd {

// Forward-declared so they can be used by splat and friends.
template <typename Child> struct base {
  __m256i value;

  // Zero constructor
  is_utf8_really_inline base() : value{__m256i()} {}

  // Conversion from SIMD register
  is_utf8_really_inline base(const __m256i _value) : value(_value) {}
  // Conversion to SIMD register
  is_utf8_really_inline operator const __m256i &() const { return this->value; }
  is_utf8_really_inline operator __m256i &() { return this->value; }
  template <endianness big_endian>
  is_utf8_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    __m256i first = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(*this));
    __m256i second = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(*this, 1));
    if (big_endian) {
      const __m256i swap = _mm256_setr_epi8(
          1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18,
          21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);
      first = _mm256_shuffle_epi8(first, swap);
      second = _mm256_shuffle_epi8(second, swap);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), first);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + 16), second);
  }
  is_utf8_really_inline void store_ascii_as_utf32(char32_t *ptr) const {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr),
                        _mm256_cvtepu8_epi32(_mm256_castsi256_si128(*this)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + 8),
                        _mm256_cvtepu8_epi32(_mm256_castsi256_si128(
                            _mm256_srli_si256(*this, 8))));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(ptr + 16),
        _mm256_cvtepu8_epi32(_mm256_extractf128_si256(*this, 1)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr + 24),
                        _mm256_cvtepu8_epi32(_mm_srli_si128(
                            _mm256_extractf128_si256(*this, 1), 8)));
  }
  // Bit operations
  is_utf8_really_inline Child operator|(const Child other) const {
    return _mm256_or_si256(*this, other);
  }
  is_utf8_really_inline Child operator&(const Child other) const {
    return _mm256_and_si256(*this, other);
  }
  is_utf8_really_inline Child operator^(const Child other) const {
    return _mm256_xor_si256(*this, other);
  }
  is_utf8_really_inline Child bit_andnot(const Child other) const {
    return _mm256_andnot_si256(other, *this);
  }
  is_utf8_really_inline Child &operator|=(const Child other) {
    auto this_cast = static_cast<Child *>(this);
    *this_cast = *this_cast | other;
    return *this_cast;
  }
  is_utf8_really_inline Child &operator&=(const Child other) {
    auto this_cast = static_cast<Child *>(this);
    *this_cast = *this_cast & other;
    return *this_cast;
  }
  is_utf8_really_inline Child &operator^=(const Child other) {
    auto this_cast = static_cast<Child *>(this);
    *this_cast = *this_cast ^ other;
    return *this_cast;
  }
};

// Forward-declared so they can be used by splat and friends.
template <typename T> struct simd8;

template <typename T, typename Mask = simd8<bool>>
struct base8 : base<simd8<T>> {
  typedef uint32_t bitmask_t;
  typedef uint64_t bitmask2_t;

  is_utf8_really_inline base8() : base<simd8<T>>() {}
  is_utf8_really_inline base8(const __m256i _value) : base<simd8<T>>(_value) {}
  is_utf8_really_inline T first() const {
    return _mm256_extract_epi8(*this, 0);
  }
  is_utf8_really_inline T last() const {
    return _mm256_extract_epi8(*this, 31);
  }
  friend is_utf8_really_inline Mask operator==(const simd8<T> lhs,
                                               const simd8<T> rhs) {
    return _mm256_cmpeq_epi8(lhs, rhs);
  }

  static const int SIZE = sizeof(base<T>::value);

  template <int N = 1>
  is_utf8_really_inline simd8<T> prev(const simd8<T> prev_chunk) const {
    return _mm256_alignr_epi8(
        *this, _mm256_permute2x128_si256(prev_chunk, *this, 0x21), 16 - N);
  }
};

// SIMD byte mask type (returned by things like eq and gt)
template <> struct simd8<bool> : base8<bool> {
  static is_utf8_really_inline simd8<bool> splat(bool _value) {
    return _mm256_set1_epi8(uint8_t(-(!!_value)));
  }

  is_utf8_really_inline simd8<bool>() : base8() {}
  is_utf8_really_inline simd8<bool>(const __m256i _value)
      : base8<bool>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd8<bool>(bool _value) : base8<bool>(splat(_value)) {}

  is_utf8_really_inline uint32_t to_bitmask() const {
    return uint32_t(_mm256_movemask_epi8(*this));
  }
  is_utf8_really_inline bool any() const {
    return !_mm256_testz_si256(*this, *this);
  }
  is_utf8_really_inline bool none() const {
    return _mm256_testz_si256(*this, *this);
  }
  is_utf8_really_inline bool all() const {
    return static_cast<uint32_t>(_mm256_movemask_epi8(*this)) == 0xFFFFFFFF;
  }
  is_utf8_really_inline simd8<bool> operator~() const { return *this ^ true; }
};

template <typename T> struct base8_numeric : base8<T> {
  static is_utf8_really_inline simd8<T> splat(T _value) {
    return _mm256_set1_epi8(_value);
  }
  static is_utf8_really_inline simd8<T> zero() {
    return _mm256_setzero_si256();
  }
  static is_utf8_really_inline simd8<T> load(const T values[32]) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
  }
  // Repeat 16 values as many times as necessary (usually for lookup tables)
  static is_utf8_really_inline simd8<T> repeat_16(T v0, T v1, T v2, T v3, T v4,
                                                  T v5, T v6, T v7, T v8, T v9,
                                                  T v10, T v11, T v12, T v13,
                                                  T v14, T v15) {
    return simd8<T>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
                    v14, v15, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                    v12, v13, v14, v15);
  }

  is_utf8_really_inline base8_numeric() : base8<T>() {}
  is_utf8_really_inline base8_numeric(const __m256i _value)
      : base8<T>(_value) {}

  // Store to array
  is_utf8_really_inline void store(T dst[32]) const {
    return _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), *this);
  }

  // Addition/subtraction are the same for signed and unsigned
  is_utf8_really_inline simd8<T> operator+(const simd8<T> other) const {
    return _mm256_add_epi8(*this, other);
  }
  is_utf8_really_inline simd8<T> operator-(const simd8<T> other) const {
    return _mm256_sub_epi8(*this, other);
  }
  is_utf8_really_inline simd8<T> &operator+=(const simd8<T> other) {
    *this = *this + other;
    return *static_cast<simd8<T> *>(this);
  }
  is_utf8_really_inline simd8<T> &operator-=(const simd8<T> other) {
    *this = *this - other;
    return *static_cast<simd8<T> *>(this);
  }

  // Override to distinguish from bool version
  is_utf8_really_inline simd8<T> operator~() const { return *this ^ 0xFFu; }

  // Perform a lookup assuming the value is between 0 and 16 (undefined behavior
  // for out of range values)
  template <typename L>
  is_utf8_really_inline simd8<L> lookup_16(simd8<L> lookup_table) const {
    return _mm256_shuffle_epi8(lookup_table, *this);
  }

  template <typename L>
  is_utf8_really_inline simd8<L>
  lookup_16(L replace0, L replace1, L replace2, L replace3, L replace4,
            L replace5, L replace6, L replace7, L replace8, L replace9,
            L replace10, L replace11, L replace12, L replace13, L replace14,
            L replace15) const {
    return lookup_16(simd8<L>::repeat_16(
        replace0, replace1, replace2, replace3, replace4, replace5, replace6,
        replace7, replace8, replace9, replace10, replace11, replace12,
        replace13, replace14, replace15));
  }
};

// Signed bytes
template <> struct simd8<int8_t> : base8_numeric<int8_t> {
  is_utf8_really_inline simd8() : base8_numeric<int8_t>() {}
  is_utf8_really_inline simd8(const __m256i _value)
      : base8_numeric<int8_t>(_value) {}

  // Splat constructor
  is_utf8_really_inline simd8(int8_t _value) : simd8(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd8(const int8_t values[32]) : simd8(load(values)) {}
  is_utf8_really_inline operator simd8<uint8_t>() const;
  // Member-by-member initialization
  is_utf8_really_inline
  simd8(int8_t v0, int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5,
        int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11,
        int8_t v12, int8_t v13, int8_t v14, int8_t v15, int8_t v16, int8_t v17,
        int8_t v18, int8_t v19, int8_t v20, int8_t v21, int8_t v22, int8_t v23,
        int8_t v24, int8_t v25, int8_t v26, int8_t v27, int8_t v28, int8_t v29,
        int8_t v30, int8_t v31)
      : simd8(_mm256_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                               v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
                               v22, v23, v24, v25, v26, v27, v28, v29, v30,
                               v31)) {}
  // Repeat 16 values as many times as necessary (usually for lookup tables)
  is_utf8_really_inline static simd8<int8_t>
  repeat_16(int8_t v0, int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5,
            int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11,
            int8_t v12, int8_t v13, int8_t v14, int8_t v15) {
    return simd8<int8_t>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
                         v13, v14, v15, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                         v10, v11, v12, v13, v14, v15);
  }
  is_utf8_really_inline bool is_ascii() const {
    return _mm256_movemask_epi8(*this) == 0;
  }
  // Order-sensitive comparisons
  is_utf8_really_inline simd8<int8_t> max_val(const simd8<int8_t> other) const {
    return _mm256_max_epi8(*this, other);
  }
  is_utf8_really_inline simd8<int8_t> min_val(const simd8<int8_t> other) const {
    return _mm256_min_epi8(*this, other);
  }
  is_utf8_really_inline simd8<bool> operator>(const simd8<int8_t> other) const {
    return _mm256_cmpgt_epi8(*this, other);
  }
  is_utf8_really_inline simd8<bool> operator<(const simd8<int8_t> other) const {
    return _mm256_cmpgt_epi8(other, *this);
  }
};

// Unsigned bytes
template <> struct simd8<uint8_t> : base8_numeric<uint8_t> {
  is_utf8_really_inline simd8() : base8_numeric<uint8_t>() {}
  is_utf8_really_inline simd8(const __m256i _value)
      : base8_numeric<uint8_t>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd8(uint8_t _value) : simd8(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd8(const uint8_t values[32]) : simd8(load(values)) {}
  // Member-by-member initialization
  is_utf8_really_inline
  simd8(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4, uint8_t v5,
        uint8_t v6, uint8_t v7, uint8_t v8, uint8_t v9, uint8_t v10,
        uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15,
        uint8_t v16, uint8_t v17, uint8_t v18, uint8_t v19, uint8_t v20,
        uint8_t v21, uint8_t v22, uint8_t v23, uint8_t v24, uint8_t v25,
        uint8_t v26, uint8_t v27, uint8_t v28, uint8_t v29, uint8_t v30,
        uint8_t v31)
      : simd8(_mm256_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                               v12, v13, v14, v15, v16, v17, v18, v19, v20, v21,
                               v22, v23, v24, v25, v26, v27, v28, v29, v30,
                               v31)) {}
  // Repeat 16 values as many times as necessary (usually for lookup tables)
  is_utf8_really_inline static simd8<uint8_t>
  repeat_16(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4,
            uint8_t v5, uint8_t v6, uint8_t v7, uint8_t v8, uint8_t v9,
            uint8_t v10, uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14,
            uint8_t v15) {
    return simd8<uint8_t>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
                          v13, v14, v15, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                          v10, v11, v12, v13, v14, v15);
  }

  // Saturated math
  is_utf8_really_inline simd8<uint8_t>
  saturating_add(const simd8<uint8_t> other) const {
    return _mm256_adds_epu8(*this, other);
  }
  is_utf8_really_inline simd8<uint8_t>
  saturating_sub(const simd8<uint8_t> other) const {
    return _mm256_subs_epu8(*this, other);
  }

  // Order-specific operations
  is_utf8_really_inline simd8<uint8_t>
  max_val(const simd8<uint8_t> other) const {
    return _mm256_max_epu8(*this, other);
  }
  is_utf8_really_inline simd8<uint8_t>
  min_val(const simd8<uint8_t> other) const {
    return _mm256_min_epu8(other, *this);
  }
  // Same as >, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd8<uint8_t>
  gt_bits(const simd8<uint8_t> other) const {
    return this->saturating_sub(other);
  }
  // Same as <, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd8<uint8_t>
  lt_bits(const simd8<uint8_t> other) const {
    return other.saturating_sub(*this);
  }
  is_utf8_really_inline simd8<bool>
  operator<=(const simd8<uint8_t> other) const {
    return other.max_val(*this) == other;
  }
  is_utf8_really_inline simd8<bool>
  operator>=(const simd8<uint8_t> other) const {
    return other.min_val(*this) == other;
  }
  is_utf8_really_inline simd8<bool>
  operator>(const simd8<uint8_t> other) const {
    return this->gt_bits(other).any_bits_set();
  }
  is_utf8_really_inline simd8<bool>
  operator<(const simd8<uint8_t> other) const {
    return this->lt_bits(other).any_bits_set();
  }

  // Bit-specific operations
  is_utf8_really_inline simd8<bool> bits_not_set() const {
    return *this == uint8_t(0);
  }
  is_utf8_really_inline simd8<bool> bits_not_set(simd8<uint8_t> bits) const {
    return (*this & bits).bits_not_set();
  }
  is_utf8_really_inline simd8<bool> any_bits_set() const {
    return ~this->bits_not_set();
  }
  is_utf8_really_inline simd8<bool> any_bits_set(simd8<uint8_t> bits) const {
    return ~this->bits_not_set(bits);
  }
  is_utf8_really_inline bool is_ascii() const {
    return _mm256_movemask_epi8(*this) == 0;
  }
  is_utf8_really_inline bool bits_not_set_anywhere() const {
    return _mm256_testz_si256(*this, *this);
  }
  is_utf8_really_inline bool any_bits_set_anywhere() const {
    return !bits_not_set_anywhere();
  }
  is_utf8_really_inline bool bits_not_set_anywhere(simd8<uint8_t> bits) const {
    return _mm256_testz_si256(*this, bits);
  }
  is_utf8_really_inline bool any_bits_set_anywhere(simd8<uint8_t> bits) const {
    return !bits_not_set_anywhere(bits);
  }
  template <int N> is_utf8_really_inline simd8<uint8_t> shr() const {
    return simd8<uint8_t>(_mm256_srli_epi16(*this, N)) & uint8_t(0xFFu >> N);
  }
  template <int N> is_utf8_really_inline simd8<uint8_t> shl() const {
    return simd8<uint8_t>(_mm256_slli_epi16(*this, N)) & uint8_t(0xFFu << N);
  }
  // Get one of the bits and make a bitmask out of it.
  // e.g. value.get_bit<7>() gets the high bit
  template <int N> is_utf8_really_inline int get_bit() const {
    return _mm256_movemask_epi8(_mm256_slli_epi16(*this, 7 - N));
  }
};
is_utf8_really_inline simd8<int8_t>::operator simd8<uint8_t>() const {
  return this->value;
}

template <typename T> struct simd8x64 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd8<T>);
  static_assert(NUM_CHUNKS == 2,
                "Haswell kernel should use two registers per 64-byte block.");
  simd8<T> chunks[NUM_CHUNKS];

  simd8x64(const simd8x64<T> &o) = delete; // no copy allowed
  simd8x64<T> &
  operator=(const simd8<T> other) = delete; // no assignment allowed
  simd8x64() = delete;                      // no default constructor allowed

  is_utf8_really_inline simd8x64(const simd8<T> chunk0, const simd8<T> chunk1)
      : chunks{chunk0, chunk1} {}
  is_utf8_really_inline simd8x64(const T *ptr)
      : chunks{simd8<T>::load(ptr),
               simd8<T>::load(ptr + sizeof(simd8<T>) / sizeof(T))} {}

  is_utf8_really_inline void store(T *ptr) const {
    this->chunks[0].store(ptr + sizeof(simd8<T>) * 0 / sizeof(T));
    this->chunks[1].store(ptr + sizeof(simd8<T>) * 1 / sizeof(T));
  }

  is_utf8_really_inline uint64_t to_bitmask() const {
    uint64_t r_lo = uint32_t(this->chunks[0].to_bitmask());
    uint64_t r_hi = this->chunks[1].to_bitmask();
    return r_lo | (r_hi << 32);
  }

  is_utf8_really_inline simd8x64<T> &operator|=(const simd8x64<T> &other) {
    this->chunks[0] |= other.chunks[0];
    this->chunks[1] |= other.chunks[1];
    return *this;
  }

  is_utf8_really_inline simd8<T> reduce_or() const {
    return this->chunks[0] | this->chunks[1];
  }

  is_utf8_really_inline bool is_ascii() const {
    return this->reduce_or().is_ascii();
  }

  template <endianness endian>
  is_utf8_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    this->chunks[0].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 0);
    this->chunks[1].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 1);
  }

  is_utf8_really_inline void store_ascii_as_utf32(char32_t *ptr) const {
    this->chunks[0].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 0);
    this->chunks[1].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 1);
  }

  is_utf8_really_inline simd8x64<T> bit_or(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<T>(this->chunks[0] | mask, this->chunks[1] | mask);
  }

  is_utf8_really_inline uint64_t eq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] == mask, this->chunks[1] == mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t eq(const simd8x64<uint8_t> &other) const {
    return simd8x64<bool>(this->chunks[0] == other.chunks[0],
                          this->chunks[1] == other.chunks[1])
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t lteq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] <= mask, this->chunks[1] <= mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t in_range(const T low, const T high) const {
    const simd8<T> mask_low = simd8<T>::splat(low);
    const simd8<T> mask_high = simd8<T>::splat(high);

    return simd8x64<bool>(
               (this->chunks[0] <= mask_high) & (this->chunks[0] >= mask_low),
               (this->chunks[1] <= mask_high) & (this->chunks[1] >= mask_low),
               (this->chunks[2] <= mask_high) & (this->chunks[2] >= mask_low),
               (this->chunks[3] <= mask_high) & (this->chunks[3] >= mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t not_in_range(const T low, const T high) const {
    const simd8<T> mask_low = simd8<T>::splat(low);
    const simd8<T> mask_high = simd8<T>::splat(high);
    return simd8x64<bool>(
               (this->chunks[0] > mask_high) | (this->chunks[0] < mask_low),
               (this->chunks[1] > mask_high) | (this->chunks[1] < mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t lt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] < mask, this->chunks[1] < mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t gt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] > mask, this->chunks[1] > mask)
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t gteq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] >= mask, this->chunks[1] >= mask)
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t gteq_unsigned(const uint8_t m) const {
    const simd8<uint8_t> mask = simd8<uint8_t>::splat(m);
    return simd8x64<bool>((simd8<uint8_t>(__m256i(this->chunks[0])) >= mask),
                          (simd8<uint8_t>(__m256i(this->chunks[1])) >= mask))
        .to_bitmask();
  }
}; // struct simd8x64<T>

#ifdef __GNUC__
#if __GNUC__ < 8
#define _mm256_set_m128i(xmm1, xmm2)                                           \
  _mm256_permute2f128_si256(_mm256_castsi128_si256(xmm1),                      \
                            _mm256_castsi128_si256(xmm2), 2)
#define _mm256_setr_m128i(xmm2, xmm1)                                          \
  _mm256_permute2f128_si256(_mm256_castsi128_si256(xmm1),                      \
                            _mm256_castsi128_si256(xmm2), 2)
#endif
#endif

template <typename T> struct simd16;

template <typename T, typename Mask = simd16<bool>>
struct base16 : base<simd16<T>> {
  using bitmask_type = uint32_t;

  is_utf8_really_inline base16() : base<simd16<T>>() {}
  is_utf8_really_inline base16(const __m256i _value)
      : base<simd16<T>>(_value) {}
  template <typename Pointer>
  is_utf8_really_inline base16(const Pointer *ptr)
      : base16(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr))) {}

  friend is_utf8_really_inline Mask operator==(const simd16<T> lhs,
                                               const simd16<T> rhs) {
    return _mm256_cmpeq_epi16(lhs, rhs);
  }

  /// the size of vector in bytes
  static const int SIZE = sizeof(base<simd16<T>>::value);

  /// the number of elements of type T a vector can hold
  static const int ELEMENTS = SIZE / sizeof(T);

  template <int N = 1>
  is_utf8_really_inline simd16<T> prev(const simd16<T> prev_chunk) const {
    return _mm256_alignr_epi8(*this, prev_chunk, 16 - N);
  }
};

// SIMD byte mask type (returned by things like eq and gt)
template <> struct simd16<bool> : base16<bool> {
  static is_utf8_really_inline simd16<bool> splat(bool _value) {
    return _mm256_set1_epi16(uint16_t(-(!!_value)));
  }

  is_utf8_really_inline simd16<bool>() : base16() {}
  is_utf8_really_inline simd16<bool>(const __m256i _value)
      : base16<bool>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd16<bool>(bool _value)
      : base16<bool>(splat(_value)) {}

  is_utf8_really_inline bitmask_type to_bitmask() const {
    return _mm256_movemask_epi8(*this);
  }
  is_utf8_really_inline bool any() const {
    return !_mm256_testz_si256(*this, *this);
  }
  is_utf8_really_inline simd16<bool> operator~() const { return *this ^ true; }
};

template <typename T> struct base16_numeric : base16<T> {
  static is_utf8_really_inline simd16<T> splat(T _value) {
    return _mm256_set1_epi16(_value);
  }
  static is_utf8_really_inline simd16<T> zero() {
    return _mm256_setzero_si256();
  }
  static is_utf8_really_inline simd16<T> load(const T values[8]) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(values));
  }

  is_utf8_really_inline base16_numeric() : base16<T>() {}
  is_utf8_really_inline base16_numeric(const __m256i _value)
      : base16<T>(_value) {}

  // Store to array
  is_utf8_really_inline void store(T dst[8]) const {
    return _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), *this);
  }

  // Override to distinguish from bool version
  is_utf8_really_inline simd16<T> operator~() const { return *this ^ 0xFFFFu; }

  // Addition/subtraction are the same for signed and unsigned
  is_utf8_really_inline simd16<T> operator+(const simd16<T> other) const {
    return _mm256_add_epi16(*this, other);
  }
  is_utf8_really_inline simd16<T> operator-(const simd16<T> other) const {
    return _mm256_sub_epi16(*this, other);
  }
  is_utf8_really_inline simd16<T> &operator+=(const simd16<T> other) {
    *this = *this + other;
    return *static_cast<simd16<T> *>(this);
  }
  is_utf8_really_inline simd16<T> &operator-=(const simd16<T> other) {
    *this = *this - other;
    return *static_cast<simd16<T> *>(this);
  }
};

// Signed words
template <> struct simd16<int16_t> : base16_numeric<int16_t> {
  is_utf8_really_inline simd16() : base16_numeric<int16_t>() {}
  is_utf8_really_inline simd16(const __m256i _value)
      : base16_numeric<int16_t>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd16(int16_t _value) : simd16(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd16(const int16_t *values) : simd16(load(values)) {}
  is_utf8_really_inline simd16(const char16_t *values)
      : simd16(load(reinterpret_cast<const int16_t *>(values))) {}
  // Order-sensitive comparisons
  is_utf8_really_inline simd16<int16_t>
  max_val(const simd16<int16_t> other) const {
    return _mm256_max_epi16(*this, other);
  }
  is_utf8_really_inline simd16<int16_t>
  min_val(const simd16<int16_t> other) const {
    return _mm256_min_epi16(*this, other);
  }
  is_utf8_really_inline simd16<bool>
  operator>(const simd16<int16_t> other) const {
    return _mm256_cmpgt_epi16(*this, other);
  }
  is_utf8_really_inline simd16<bool>
  operator<(const simd16<int16_t> other) const {
    return _mm256_cmpgt_epi16(other, *this);
  }
};

// Unsigned words
template <> struct simd16<uint16_t> : base16_numeric<uint16_t> {
  is_utf8_really_inline simd16() : base16_numeric<uint16_t>() {}
  is_utf8_really_inline simd16(const __m256i _value)
      : base16_numeric<uint16_t>(_value) {}

  // Splat constructor
  is_utf8_really_inline simd16(uint16_t _value) : simd16(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd16(const uint16_t *values) : simd16(load(values)) {}
  is_utf8_really_inline simd16(const char16_t *values)
      : simd16(load(reinterpret_cast<const uint16_t *>(values))) {}

  // Saturated math
  is_utf8_really_inline simd16<uint16_t>
  saturating_add(const simd16<uint16_t> other) const {
    return _mm256_adds_epu16(*this, other);
  }
  is_utf8_really_inline simd16<uint16_t>
  saturating_sub(const simd16<uint16_t> other) const {
    return _mm256_subs_epu16(*this, other);
  }

  // Order-specific operations
  is_utf8_really_inline simd16<uint16_t>
  max_val(const simd16<uint16_t> other) const {
    return _mm256_max_epu16(*this, other);
  }
  is_utf8_really_inline simd16<uint16_t>
  min_val(const simd16<uint16_t> other) const {
    return _mm256_min_epu16(*this, other);
  }
  // Same as >, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd16<uint16_t>
  gt_bits(const simd16<uint16_t> other) const {
    return this->saturating_sub(other);
  }
  // Same as <, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd16<uint16_t>
  lt_bits(const simd16<uint16_t> other) const {
    return other.saturating_sub(*this);
  }
  is_utf8_really_inline simd16<bool>
  operator<=(const simd16<uint16_t> other) const {
    return other.max_val(*this) == other;
  }
  is_utf8_really_inline simd16<bool>
  operator>=(const simd16<uint16_t> other) const {
    return other.min_val(*this) == other;
  }
  is_utf8_really_inline simd16<bool>
  operator>(const simd16<uint16_t> other) const {
    return this->gt_bits(other).any_bits_set();
  }
  is_utf8_really_inline simd16<bool>
  operator<(const simd16<uint16_t> other) const {
    return this->gt_bits(other).any_bits_set();
  }

  // Bit-specific operations
  is_utf8_really_inline simd16<bool> bits_not_set() const {
    return *this == uint16_t(0);
  }
  is_utf8_really_inline simd16<bool> bits_not_set(simd16<uint16_t> bits) const {
    return (*this & bits).bits_not_set();
  }
  is_utf8_really_inline simd16<bool> any_bits_set() const {
    return ~this->bits_not_set();
  }
  is_utf8_really_inline simd16<bool> any_bits_set(simd16<uint16_t> bits) const {
    return ~this->bits_not_set(bits);
  }

  is_utf8_really_inline bool bits_not_set_anywhere() const {
    return _mm256_testz_si256(*this, *this);
  }
  is_utf8_really_inline bool any_bits_set_anywhere() const {
    return !bits_not_set_anywhere();
  }
  is_utf8_really_inline bool
  bits_not_set_anywhere(simd16<uint16_t> bits) const {
    return _mm256_testz_si256(*this, bits);
  }
  is_utf8_really_inline bool
  any_bits_set_anywhere(simd16<uint16_t> bits) const {
    return !bits_not_set_anywhere(bits);
  }
  template <int N> is_utf8_really_inline simd16<uint16_t> shr() const {
    return simd16<uint16_t>(_mm256_srli_epi16(*this, N));
  }
  template <int N> is_utf8_really_inline simd16<uint16_t> shl() const {
    return simd16<uint16_t>(_mm256_slli_epi16(*this, N));
  }
  // Get one of the bits and make a bitmask out of it.
  // e.g. value.get_bit<7>() gets the high bit
  template <int N> is_utf8_really_inline int get_bit() const {
    return _mm256_movemask_epi8(_mm256_slli_epi16(*this, 15 - N));
  }

  // Change the endianness
  is_utf8_really_inline simd16<uint16_t> swap_bytes() const {
    const __m256i swap = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18,
        21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);
    return _mm256_shuffle_epi8(*this, swap);
  }

  // Pack with the unsigned saturation two uint16_t words into single uint8_t
  // vector
  static is_utf8_really_inline simd8<uint8_t> pack(const simd16<uint16_t> &v0,
                                                   const simd16<uint16_t> &v1) {
    // Note: the AVX2 variant of pack operates on 128-bit lanes, thus
    //       we have to shuffle lanes in order to produce bytes in the
    //       correct order.

    // get the 0th lanes
    const __m128i lo_0 = _mm256_extracti128_si256(v0, 0);
    const __m128i lo_1 = _mm256_extracti128_si256(v1, 0);

    // get the 1st lanes
    const __m128i hi_0 = _mm256_extracti128_si256(v0, 1);
    const __m128i hi_1 = _mm256_extracti128_si256(v1, 1);

    // build new vectors (shuffle lanes)
    const __m256i t0 = _mm256_set_m128i(lo_1, lo_0);
    const __m256i t1 = _mm256_set_m128i(hi_1, hi_0);

    // pack words in linear order from v0 and v1
    return _mm256_packus_epi16(t0, t1);
  }
};

template <typename T> struct simd16x32 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd16<T>);
  static_assert(NUM_CHUNKS == 2,
                "Haswell kernel should use two registers per 64-byte block.");
  simd16<T> chunks[NUM_CHUNKS];

  simd16x32(const simd16x32<T> &o) = delete; // no copy allowed
  simd16x32<T> &
  operator=(const simd16<T> other) = delete; // no assignment allowed
  simd16x32() = delete;                      // no default constructor allowed

  is_utf8_really_inline simd16x32(const simd16<T> chunk0,
                                  const simd16<T> chunk1)
      : chunks{chunk0, chunk1} {}
  is_utf8_really_inline simd16x32(const T *ptr)
      : chunks{simd16<T>::load(ptr),
               simd16<T>::load(ptr + sizeof(simd16<T>) / sizeof(T))} {}

  is_utf8_really_inline void store(T *ptr) const {
    this->chunks[0].store(ptr + sizeof(simd16<T>) * 0 / sizeof(T));
    this->chunks[1].store(ptr + sizeof(simd16<T>) * 1 / sizeof(T));
  }

  is_utf8_really_inline uint64_t to_bitmask() const {
    uint64_t r_lo = uint32_t(this->chunks[0].to_bitmask());
    uint64_t r_hi = this->chunks[1].to_bitmask();
    return r_lo | (r_hi << 32);
  }

  is_utf8_really_inline simd16<T> reduce_or() const {
    return this->chunks[0] | this->chunks[1];
  }

  is_utf8_really_inline bool is_ascii() const {
    return this->reduce_or().is_ascii();
  }

  is_utf8_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    this->chunks[0].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 0);
    this->chunks[1].store_ascii_as_utf16(ptr + sizeof(simd16<T>));
  }

  is_utf8_really_inline simd16x32<T> bit_or(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<T>(this->chunks[0] | mask, this->chunks[1] | mask);
  }

  is_utf8_really_inline void swap_bytes() {
    this->chunks[0] = this->chunks[0].swap_bytes();
    this->chunks[1] = this->chunks[1].swap_bytes();
  }

  is_utf8_really_inline uint64_t eq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] == mask, this->chunks[1] == mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t eq(const simd16x32<uint16_t> &other) const {
    return simd16x32<bool>(this->chunks[0] == other.chunks[0],
                           this->chunks[1] == other.chunks[1])
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t lteq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] <= mask, this->chunks[1] <= mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t in_range(const T low, const T high) const {
    const simd16<T> mask_low = simd16<T>::splat(low);
    const simd16<T> mask_high = simd16<T>::splat(high);

    return simd16x32<bool>(
               (this->chunks[0] <= mask_high) & (this->chunks[0] >= mask_low),
               (this->chunks[1] <= mask_high) & (this->chunks[1] >= mask_low),
               (this->chunks[2] <= mask_high) & (this->chunks[2] >= mask_low),
               (this->chunks[3] <= mask_high) & (this->chunks[3] >= mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t not_in_range(const T low, const T high) const {
    const simd16<T> mask_low = simd16<T>::splat(static_cast<T>(low - 1));
    const simd16<T> mask_high = simd16<T>::splat(static_cast<T>(high + 1));
    return simd16x32<bool>(
               (this->chunks[0] >= mask_high) | (this->chunks[0] <= mask_low),
               (this->chunks[1] >= mask_high) | (this->chunks[1] <= mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t lt(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] < mask, this->chunks[1] < mask)
        .to_bitmask();
  }
}; // struct simd16x32<T>

} // namespace simd

} // unnamed namespace
} // namespace haswell
} // namespace is_utf8_internals

#endif // IS_UTF8_HASWELL_SIMD_H

IS_UTF8_UNTARGET_REGION

#if IS_UTF8_GCC11ORMORE // workaround for
                        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
#pragma GCC diagnostic pop
#endif // end of workaround

#endif // IS_UTF8_IMPLEMENTATION_HASWELL
#endif // IS_UTF8_HASWELL_COMMON_H

#ifndef IS_UTF8_WESTMERE_H
#define IS_UTF8_WESTMERE_H

#ifdef IS_UTF8_FALLBACK_H
#error "westmere.h must be included before fallback.h"
#endif

// Default Westmere to on if this is x86-64, unless we'll always select Haswell.
#ifndef IS_UTF8_IMPLEMENTATION_WESTMERE
//
// You do not want to set it to (IS_UTF8_IS_X86_64 && !IS_UTF8_REQUIRES_HASWELL)
// because you want to rely on runtime dispatch!
//
#if IS_UTF8_CAN_ALWAYS_RUN_ICELAKE || IS_UTF8_CAN_ALWAYS_RUN_HASWELL
#define IS_UTF8_IMPLEMENTATION_WESTMERE 0
#else
#define IS_UTF8_IMPLEMENTATION_WESTMERE (IS_UTF8_IS_X86_64)
#endif

#endif

#if IS_UTF8_IMPLEMENTATION_WESTMERE && IS_UTF8_IS_X86_64 && __SSE4_2__ &&  __PCLMUL__
#define IS_UTF8_CAN_ALWAYS_RUN_WESTMERE 1
#else
#define IS_UTF8_CAN_ALWAYS_RUN_WESTMERE 0
#endif

#if IS_UTF8_IMPLEMENTATION_WESTMERE

#define IS_UTF8_TARGET_WESTMERE IS_UTF8_TARGET_REGION("sse4.2,pclmul")

namespace is_utf8_internals {
/**
 * Implementation for Westmere (Intel SSE4.2).
 */
namespace westmere {} // namespace westmere
} // namespace is_utf8_internals

//
// These two need to be included outside IS_UTF8_TARGET_REGION
//

#ifndef IS_UTF8_WESTMERE_IMPLEMENTATION_H
#define IS_UTF8_WESTMERE_IMPLEMENTATION_H

// The constructor may be executed on any host, so we take care not to use
// IS_UTF8_TARGET_REGION
namespace is_utf8_internals {
namespace westmere {

class implementation final : public is_utf8_internals::implementation {
public:
  is_utf8_really_inline implementation()
      : is_utf8_internals::implementation(
            "westmere", "Intel/AMD SSE4.2",
            internal::instruction_set::SSE42 |
                internal::instruction_set::PCLMULQDQ) {}
  is_utf8_warn_unused bool validate_utf8(const char *buf,
                                         size_t len) const noexcept final;
};

} // namespace westmere
} // namespace is_utf8_internals

#endif // IS_UTF8_WESTMERE_IMPLEMENTATION_H

#ifndef IS_UTF8_WESTMERE_INTRINSICS_H
#define IS_UTF8_WESTMERE_INTRINSICS_H

#ifdef IS_UTF8_VISUAL_STUDIO
// under clang within visual studio, this will include <x86intrin.h>
#include <intrin.h> // visual studio or clang
#else

#if IS_UTF8_GCC11ORMORE
// We should not get warnings while including <x86intrin.h> yet we do
// under some versions of GCC.
// If the x86intrin.h header has uninitialized values that are problematic,
// it is a GCC issue, we want to ignore these warnings.
IS_UTF8_DISABLE_GCC_WARNING(-Wuninitialized)
#endif

#include <x86intrin.h> // elsewhere

#if IS_UTF8_GCC11ORMORE
// cancels the suppression of the -Wuninitialized
IS_UTF8_POP_DISABLE_WARNINGS
#endif

#endif // IS_UTF8_VISUAL_STUDIO

#ifdef IS_UTF8_CLANG_VISUAL_STUDIO
/**
 * You are not supposed, normally, to include these
 * headers directly. Instead you should either include intrin.h
 * or x86intrin.h. However, when compiling with clang
 * under Windows (i.e., when _MSC_VER is set), these headers
 * only get included *if* the corresponding features are detected
 * from macros:
 */
#include <smmintrin.h> // for _mm_alignr_epi8
#include <wmmintrin.h> // for  _mm_clmulepi64_si128
#endif

#endif // IS_UTF8_WESTMERE_INTRINSICS_H

//
// The rest need to be inside the region
//

// redefining IS_UTF8_IMPLEMENTATION to "westmere"
// #define IS_UTF8_IMPLEMENTATION westmere
IS_UTF8_TARGET_WESTMERE

// Declarations

#ifndef IS_UTF8_WESTMERE_SIMD_H
#define IS_UTF8_WESTMERE_SIMD_H

namespace is_utf8_internals {
namespace westmere {
namespace {
namespace simd {

template <typename Child> struct base {
  __m128i value;

  // Zero constructor
  is_utf8_really_inline base() : value{__m128i()} {}

  // Conversion from SIMD register
  is_utf8_really_inline base(const __m128i _value) : value(_value) {}
  // Conversion to SIMD register
  is_utf8_really_inline operator const __m128i &() const { return this->value; }
  is_utf8_really_inline operator __m128i &() { return this->value; }
  template <endianness big_endian>
  is_utf8_really_inline void store_ascii_as_utf16(char16_t *p) const {
    __m128i first = _mm_cvtepu8_epi16(*this);
    __m128i second = _mm_cvtepu8_epi16(_mm_srli_si128(*this, 8));
    if (big_endian) {
      const __m128i swap =
          _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
      first = _mm_shuffle_epi8(first, swap);
      second = _mm_shuffle_epi8(second, swap);
    }
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), first);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p + 8), second);
  }
  is_utf8_really_inline void store_ascii_as_utf32(char32_t *p) const {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p), _mm_cvtepu8_epi32(*this));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p + 4),
                     _mm_cvtepu8_epi32(_mm_srli_si128(*this, 4)));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p + 8),
                     _mm_cvtepu8_epi32(_mm_srli_si128(*this, 8)));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(p + 12),
                     _mm_cvtepu8_epi32(_mm_srli_si128(*this, 12)));
  }
  // Bit operations
  is_utf8_really_inline Child operator|(const Child other) const {
    return _mm_or_si128(*this, other);
  }
  is_utf8_really_inline Child operator&(const Child other) const {
    return _mm_and_si128(*this, other);
  }
  is_utf8_really_inline Child operator^(const Child other) const {
    return _mm_xor_si128(*this, other);
  }
  is_utf8_really_inline Child bit_andnot(const Child other) const {
    return _mm_andnot_si128(other, *this);
  }
  is_utf8_really_inline Child &operator|=(const Child other) {
    auto this_cast = static_cast<Child *>(this);
    *this_cast = *this_cast | other;
    return *this_cast;
  }
  is_utf8_really_inline Child &operator&=(const Child other) {
    auto this_cast = static_cast<Child *>(this);
    *this_cast = *this_cast & other;
    return *this_cast;
  }
  is_utf8_really_inline Child &operator^=(const Child other) {
    auto this_cast = static_cast<Child *>(this);
    *this_cast = *this_cast ^ other;
    return *this_cast;
  }
};

// Forward-declared so they can be used by splat and friends.
template <typename T> struct simd8;

template <typename T, typename Mask = simd8<bool>>
struct base8 : base<simd8<T>> {
  typedef uint16_t bitmask_t;
  typedef uint32_t bitmask2_t;

  is_utf8_really_inline T first() const { return _mm_extract_epi8(*this, 0); }
  is_utf8_really_inline T last() const { return _mm_extract_epi8(*this, 15); }
  is_utf8_really_inline base8() : base<simd8<T>>() {}
  is_utf8_really_inline base8(const __m128i _value) : base<simd8<T>>(_value) {}

  friend is_utf8_really_inline Mask operator==(const simd8<T> lhs,
                                               const simd8<T> rhs) {
    return _mm_cmpeq_epi8(lhs, rhs);
  }

  static const int SIZE = sizeof(base<simd8<T>>::value);

  template <int N = 1>
  is_utf8_really_inline simd8<T> prev(const simd8<T> prev_chunk) const {
    return _mm_alignr_epi8(*this, prev_chunk, 16 - N);
  }
};

// SIMD byte mask type (returned by things like eq and gt)
template <> struct simd8<bool> : base8<bool> {
  static is_utf8_really_inline simd8<bool> splat(bool _value) {
    return _mm_set1_epi8(uint8_t(-(!!_value)));
  }

  is_utf8_really_inline simd8<bool>() : base8() {}
  is_utf8_really_inline simd8<bool>(const __m128i _value)
      : base8<bool>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd8<bool>(bool _value) : base8<bool>(splat(_value)) {}

  is_utf8_really_inline int to_bitmask() const {
    return _mm_movemask_epi8(*this);
  }
  is_utf8_really_inline bool any() const {
    return !_mm_testz_si128(*this, *this);
  }
  is_utf8_really_inline bool none() const {
    return _mm_testz_si128(*this, *this);
  }
  is_utf8_really_inline bool all() const {
    return _mm_movemask_epi8(*this) == 0xFFFF;
  }
  is_utf8_really_inline simd8<bool> operator~() const { return *this ^ true; }
};

template <typename T> struct base8_numeric : base8<T> {
  static is_utf8_really_inline simd8<T> splat(T _value) {
    return _mm_set1_epi8(_value);
  }
  static is_utf8_really_inline simd8<T> zero() { return _mm_setzero_si128(); }
  static is_utf8_really_inline simd8<T> load(const T values[16]) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(values));
  }
  // Repeat 16 values as many times as necessary (usually for lookup tables)
  static is_utf8_really_inline simd8<T> repeat_16(T v0, T v1, T v2, T v3, T v4,
                                                  T v5, T v6, T v7, T v8, T v9,
                                                  T v10, T v11, T v12, T v13,
                                                  T v14, T v15) {
    return simd8<T>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13,
                    v14, v15);
  }

  is_utf8_really_inline base8_numeric() : base8<T>() {}
  is_utf8_really_inline base8_numeric(const __m128i _value)
      : base8<T>(_value) {}

  // Store to array
  is_utf8_really_inline void store(T dst[16]) const {
    return _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), *this);
  }

  // Override to distinguish from bool version
  is_utf8_really_inline simd8<T> operator~() const { return *this ^ 0xFFu; }

  // Addition/subtraction are the same for signed and unsigned
  is_utf8_really_inline simd8<T> operator+(const simd8<T> other) const {
    return _mm_add_epi8(*this, other);
  }
  is_utf8_really_inline simd8<T> operator-(const simd8<T> other) const {
    return _mm_sub_epi8(*this, other);
  }
  is_utf8_really_inline simd8<T> &operator+=(const simd8<T> other) {
    *this = *this + other;
    return *static_cast<simd8<T> *>(this);
  }
  is_utf8_really_inline simd8<T> &operator-=(const simd8<T> other) {
    *this = *this - other;
    return *static_cast<simd8<T> *>(this);
  }

  // Perform a lookup assuming the value is between 0 and 16 (undefined behavior
  // for out of range values)
  template <typename L>
  is_utf8_really_inline simd8<L> lookup_16(simd8<L> lookup_table) const {
    return _mm_shuffle_epi8(lookup_table, *this);
  }

  template <typename L>
  is_utf8_really_inline simd8<L>
  lookup_16(L replace0, L replace1, L replace2, L replace3, L replace4,
            L replace5, L replace6, L replace7, L replace8, L replace9,
            L replace10, L replace11, L replace12, L replace13, L replace14,
            L replace15) const {
    return lookup_16(simd8<L>::repeat_16(
        replace0, replace1, replace2, replace3, replace4, replace5, replace6,
        replace7, replace8, replace9, replace10, replace11, replace12,
        replace13, replace14, replace15));
  }
};

// Signed bytes
template <> struct simd8<int8_t> : base8_numeric<int8_t> {
  is_utf8_really_inline simd8() : base8_numeric<int8_t>() {}
  is_utf8_really_inline simd8(const __m128i _value)
      : base8_numeric<int8_t>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd8(int8_t _value) : simd8(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd8(const int8_t *values) : simd8(load(values)) {}
  // Member-by-member initialization
  is_utf8_really_inline simd8(int8_t v0, int8_t v1, int8_t v2, int8_t v3,
                              int8_t v4, int8_t v5, int8_t v6, int8_t v7,
                              int8_t v8, int8_t v9, int8_t v10, int8_t v11,
                              int8_t v12, int8_t v13, int8_t v14, int8_t v15)
      : simd8(_mm_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                            v12, v13, v14, v15)) {}
  // Repeat 16 values as many times as necessary (usually for lookup tables)
  is_utf8_really_inline static simd8<int8_t>
  repeat_16(int8_t v0, int8_t v1, int8_t v2, int8_t v3, int8_t v4, int8_t v5,
            int8_t v6, int8_t v7, int8_t v8, int8_t v9, int8_t v10, int8_t v11,
            int8_t v12, int8_t v13, int8_t v14, int8_t v15) {
    return simd8<int8_t>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
                         v13, v14, v15);
  }
  is_utf8_really_inline operator simd8<uint8_t>() const;
  is_utf8_really_inline bool is_ascii() const {
    return _mm_movemask_epi8(*this) == 0;
  }

  // Order-sensitive comparisons
  is_utf8_really_inline simd8<int8_t> max_val(const simd8<int8_t> other) const {
    return _mm_max_epi8(*this, other);
  }
  is_utf8_really_inline simd8<int8_t> min_val(const simd8<int8_t> other) const {
    return _mm_min_epi8(*this, other);
  }
  is_utf8_really_inline simd8<bool> operator>(const simd8<int8_t> other) const {
    return _mm_cmpgt_epi8(*this, other);
  }
  is_utf8_really_inline simd8<bool> operator<(const simd8<int8_t> other) const {
    return _mm_cmpgt_epi8(other, *this);
  }
};

// Unsigned bytes
template <> struct simd8<uint8_t> : base8_numeric<uint8_t> {
  is_utf8_really_inline simd8() : base8_numeric<uint8_t>() {}
  is_utf8_really_inline simd8(const __m128i _value)
      : base8_numeric<uint8_t>(_value) {}

  // Splat constructor
  is_utf8_really_inline simd8(uint8_t _value) : simd8(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd8(const uint8_t *values) : simd8(load(values)) {}
  // Member-by-member initialization
  is_utf8_really_inline
  simd8(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4, uint8_t v5,
        uint8_t v6, uint8_t v7, uint8_t v8, uint8_t v9, uint8_t v10,
        uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14, uint8_t v15)
      : simd8(_mm_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                            v12, v13, v14, v15)) {}
  // Repeat 16 values as many times as necessary (usually for lookup tables)
  is_utf8_really_inline static simd8<uint8_t>
  repeat_16(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3, uint8_t v4,
            uint8_t v5, uint8_t v6, uint8_t v7, uint8_t v8, uint8_t v9,
            uint8_t v10, uint8_t v11, uint8_t v12, uint8_t v13, uint8_t v14,
            uint8_t v15) {
    return simd8<uint8_t>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12,
                          v13, v14, v15);
  }

  // Saturated math
  is_utf8_really_inline simd8<uint8_t>
  saturating_add(const simd8<uint8_t> other) const {
    return _mm_adds_epu8(*this, other);
  }
  is_utf8_really_inline simd8<uint8_t>
  saturating_sub(const simd8<uint8_t> other) const {
    return _mm_subs_epu8(*this, other);
  }

  // Order-specific operations
  is_utf8_really_inline simd8<uint8_t>
  max_val(const simd8<uint8_t> other) const {
    return _mm_max_epu8(*this, other);
  }
  is_utf8_really_inline simd8<uint8_t>
  min_val(const simd8<uint8_t> other) const {
    return _mm_min_epu8(*this, other);
  }
  // Same as >, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd8<uint8_t>
  gt_bits(const simd8<uint8_t> other) const {
    return this->saturating_sub(other);
  }
  // Same as <, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd8<uint8_t>
  lt_bits(const simd8<uint8_t> other) const {
    return other.saturating_sub(*this);
  }
  is_utf8_really_inline simd8<bool>
  operator<=(const simd8<uint8_t> other) const {
    return other.max_val(*this) == other;
  }
  is_utf8_really_inline simd8<bool>
  operator>=(const simd8<uint8_t> other) const {
    return other.min_val(*this) == other;
  }
  is_utf8_really_inline simd8<bool>
  operator>(const simd8<uint8_t> other) const {
    return this->gt_bits(other).any_bits_set();
  }
  is_utf8_really_inline simd8<bool>
  operator<(const simd8<uint8_t> other) const {
    return this->gt_bits(other).any_bits_set();
  }

  // Bit-specific operations
  is_utf8_really_inline simd8<bool> bits_not_set() const {
    return *this == uint8_t(0);
  }
  is_utf8_really_inline simd8<bool> bits_not_set(simd8<uint8_t> bits) const {
    return (*this & bits).bits_not_set();
  }
  is_utf8_really_inline simd8<bool> any_bits_set() const {
    return ~this->bits_not_set();
  }
  is_utf8_really_inline simd8<bool> any_bits_set(simd8<uint8_t> bits) const {
    return ~this->bits_not_set(bits);
  }
  is_utf8_really_inline bool is_ascii() const {
    return _mm_movemask_epi8(*this) == 0;
  }

  is_utf8_really_inline bool bits_not_set_anywhere() const {
    return _mm_testz_si128(*this, *this);
  }
  is_utf8_really_inline bool any_bits_set_anywhere() const {
    return !bits_not_set_anywhere();
  }
  is_utf8_really_inline bool bits_not_set_anywhere(simd8<uint8_t> bits) const {
    return _mm_testz_si128(*this, bits);
  }
  is_utf8_really_inline bool any_bits_set_anywhere(simd8<uint8_t> bits) const {
    return !bits_not_set_anywhere(bits);
  }
  template <int N> is_utf8_really_inline simd8<uint8_t> shr() const {
    return simd8<uint8_t>(_mm_srli_epi16(*this, N)) & uint8_t(0xFFu >> N);
  }
  template <int N> is_utf8_really_inline simd8<uint8_t> shl() const {
    return simd8<uint8_t>(_mm_slli_epi16(*this, N)) & uint8_t(0xFFu << N);
  }
  // Get one of the bits and make a bitmask out of it.
  // e.g. value.get_bit<7>() gets the high bit
  template <int N> is_utf8_really_inline int get_bit() const {
    return _mm_movemask_epi8(_mm_slli_epi16(*this, 7 - N));
  }
};
is_utf8_really_inline simd8<int8_t>::operator simd8<uint8_t>() const {
  return this->value;
}

// Unsigned bytes
template <> struct simd8<uint16_t> : base<uint16_t> {
  static is_utf8_really_inline simd8<uint16_t> splat(uint16_t _value) {
    return _mm_set1_epi16(_value);
  }
  static is_utf8_really_inline simd8<uint16_t> load(const uint16_t values[8]) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(values));
  }

  is_utf8_really_inline simd8() : base<uint16_t>() {}
  is_utf8_really_inline simd8(const __m128i _value) : base<uint16_t>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd8(uint16_t _value) : simd8(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd8(const uint16_t *values) : simd8(load(values)) {}
  // Member-by-member initialization
  is_utf8_really_inline simd8(uint16_t v0, uint16_t v1, uint16_t v2,
                              uint16_t v3, uint16_t v4, uint16_t v5,
                              uint16_t v6, uint16_t v7)
      : simd8(_mm_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7)) {}

  // Saturated math
  is_utf8_really_inline simd8<uint16_t>
  saturating_add(const simd8<uint16_t> other) const {
    return _mm_adds_epu16(*this, other);
  }
  is_utf8_really_inline simd8<uint16_t>
  saturating_sub(const simd8<uint16_t> other) const {
    return _mm_subs_epu16(*this, other);
  }

  // Order-specific operations
  is_utf8_really_inline simd8<uint16_t>
  max_val(const simd8<uint16_t> other) const {
    return _mm_max_epu16(*this, other);
  }
  is_utf8_really_inline simd8<uint16_t>
  min_val(const simd8<uint16_t> other) const {
    return _mm_min_epu16(*this, other);
  }
  // Same as >, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd8<uint16_t>
  gt_bits(const simd8<uint16_t> other) const {
    return this->saturating_sub(other);
  }
  // Same as <, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd8<uint16_t>
  lt_bits(const simd8<uint16_t> other) const {
    return other.saturating_sub(*this);
  }
  is_utf8_really_inline simd8<bool>
  operator<=(const simd8<uint16_t> other) const {
    return other.max_val(*this) == other;
  }
  is_utf8_really_inline simd8<bool>
  operator>=(const simd8<uint16_t> other) const {
    return other.min_val(*this) == other;
  }
  is_utf8_really_inline simd8<bool>
  operator==(const simd8<uint16_t> other) const {
    return _mm_cmpeq_epi16(*this, other);
  }
  is_utf8_really_inline simd8<bool>
  operator&(const simd8<uint16_t> other) const {
    return _mm_and_si128(*this, other);
  }
  is_utf8_really_inline simd8<bool>
  operator|(const simd8<uint16_t> other) const {
    return _mm_or_si128(*this, other);
  }

  // Bit-specific operations
  is_utf8_really_inline simd8<bool> bits_not_set() const {
    return *this == uint16_t(0);
  }
  is_utf8_really_inline simd8<bool> any_bits_set() const {
    return ~this->bits_not_set();
  }

  is_utf8_really_inline bool bits_not_set_anywhere() const {
    return _mm_testz_si128(*this, *this);
  }
  is_utf8_really_inline bool any_bits_set_anywhere() const {
    return !bits_not_set_anywhere();
  }
  is_utf8_really_inline bool bits_not_set_anywhere(simd8<uint16_t> bits) const {
    return _mm_testz_si128(*this, bits);
  }
  is_utf8_really_inline bool any_bits_set_anywhere(simd8<uint16_t> bits) const {
    return !bits_not_set_anywhere(bits);
  }
};
template <typename T> struct simd8x64 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd8<T>);
  static_assert(NUM_CHUNKS == 4,
                "Westmere kernel should use four registers per 64-byte block.");
  simd8<T> chunks[NUM_CHUNKS];

  simd8x64(const simd8x64<T> &o) = delete; // no copy allowed
  simd8x64<T> &
  operator=(const simd8<T> other) = delete; // no assignment allowed
  simd8x64() = delete;                      // no default constructor allowed

  is_utf8_really_inline simd8x64(const simd8<T> chunk0, const simd8<T> chunk1,
                                 const simd8<T> chunk2, const simd8<T> chunk3)
      : chunks{chunk0, chunk1, chunk2, chunk3} {}
  is_utf8_really_inline simd8x64(const T *ptr)
      : chunks{simd8<T>::load(ptr),
               simd8<T>::load(ptr + sizeof(simd8<T>) / sizeof(T)),
               simd8<T>::load(ptr + 2 * sizeof(simd8<T>) / sizeof(T)),
               simd8<T>::load(ptr + 3 * sizeof(simd8<T>) / sizeof(T))} {}

  is_utf8_really_inline void store(T *ptr) const {
    this->chunks[0].store(ptr + sizeof(simd8<T>) * 0 / sizeof(T));
    this->chunks[1].store(ptr + sizeof(simd8<T>) * 1 / sizeof(T));
    this->chunks[2].store(ptr + sizeof(simd8<T>) * 2 / sizeof(T));
    this->chunks[3].store(ptr + sizeof(simd8<T>) * 3 / sizeof(T));
  }

  is_utf8_really_inline simd8x64<T> &operator|=(const simd8x64<T> &other) {
    this->chunks[0] |= other.chunks[0];
    this->chunks[1] |= other.chunks[1];
    this->chunks[2] |= other.chunks[2];
    this->chunks[3] |= other.chunks[3];
    return *this;
  }

  is_utf8_really_inline simd8<T> reduce_or() const {
    return (this->chunks[0] | this->chunks[1]) |
           (this->chunks[2] | this->chunks[3]);
  }

  is_utf8_really_inline bool is_ascii() const {
    return this->reduce_or().is_ascii();
  }

  template <endianness endian>
  is_utf8_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    this->chunks[0].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 0);
    this->chunks[1].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 1);
    this->chunks[2].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 2);
    this->chunks[3].template store_ascii_as_utf16<endian>(ptr +
                                                          sizeof(simd8<T>) * 3);
  }

  is_utf8_really_inline void store_ascii_as_utf32(char32_t *ptr) const {
    this->chunks[0].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 0);
    this->chunks[1].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 1);
    this->chunks[2].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 2);
    this->chunks[3].store_ascii_as_utf32(ptr + sizeof(simd8<T>) * 3);
  }

  is_utf8_really_inline uint64_t to_bitmask() const {
    uint64_t r0 = uint32_t(this->chunks[0].to_bitmask());
    uint64_t r1 = this->chunks[1].to_bitmask();
    uint64_t r2 = this->chunks[2].to_bitmask();
    uint64_t r3 = this->chunks[3].to_bitmask();
    return r0 | (r1 << 16) | (r2 << 32) | (r3 << 48);
  }

  is_utf8_really_inline uint64_t eq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] == mask, this->chunks[1] == mask,
                          this->chunks[2] == mask, this->chunks[3] == mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t eq(const simd8x64<uint8_t> &other) const {
    return simd8x64<bool>(this->chunks[0] == other.chunks[0],
                          this->chunks[1] == other.chunks[1],
                          this->chunks[2] == other.chunks[2],
                          this->chunks[3] == other.chunks[3])
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t lteq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] <= mask, this->chunks[1] <= mask,
                          this->chunks[2] <= mask, this->chunks[3] <= mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t in_range(const T low, const T high) const {
    const simd8<T> mask_low = simd8<T>::splat(low);
    const simd8<T> mask_high = simd8<T>::splat(high);

    return simd8x64<bool>(
               (this->chunks[0] <= mask_high) & (this->chunks[0] >= mask_low),
               (this->chunks[1] <= mask_high) & (this->chunks[1] >= mask_low),
               (this->chunks[2] <= mask_high) & (this->chunks[2] >= mask_low),
               (this->chunks[3] <= mask_high) & (this->chunks[3] >= mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t not_in_range(const T low, const T high) const {
    const simd8<T> mask_low = simd8<T>::splat(low - 1);
    const simd8<T> mask_high = simd8<T>::splat(high + 1);
    return simd8x64<bool>(
               (this->chunks[0] >= mask_high) | (this->chunks[0] <= mask_low),
               (this->chunks[1] >= mask_high) | (this->chunks[1] <= mask_low),
               (this->chunks[2] >= mask_high) | (this->chunks[2] <= mask_low),
               (this->chunks[3] >= mask_high) | (this->chunks[3] <= mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t lt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] < mask, this->chunks[1] < mask,
                          this->chunks[2] < mask, this->chunks[3] < mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t gt(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] > mask, this->chunks[1] > mask,
                          this->chunks[2] > mask, this->chunks[3] > mask)
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t gteq(const T m) const {
    const simd8<T> mask = simd8<T>::splat(m);
    return simd8x64<bool>(this->chunks[0] >= mask, this->chunks[1] >= mask,
                          this->chunks[2] >= mask, this->chunks[3] >= mask)
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t gteq_unsigned(const uint8_t m) const {
    const simd8<uint8_t> mask = simd8<uint8_t>::splat(m);
    return simd8x64<bool>(simd8<uint8_t>(__m128i(this->chunks[0])) >= mask,
                          simd8<uint8_t>(__m128i(this->chunks[1])) >= mask,
                          simd8<uint8_t>(__m128i(this->chunks[2])) >= mask,
                          simd8<uint8_t>(__m128i(this->chunks[3])) >= mask)
        .to_bitmask();
  }
}; // struct simd8x64<T>

template <typename T> struct simd16;

template <typename T, typename Mask = simd16<bool>>
struct base16 : base<simd16<T>> {
  typedef uint16_t bitmask_t;
  typedef uint32_t bitmask2_t;

  is_utf8_really_inline base16() : base<simd16<T>>() {}
  is_utf8_really_inline base16(const __m128i _value)
      : base<simd16<T>>(_value) {}
  template <typename Pointer>
  is_utf8_really_inline base16(const Pointer *ptr)
      : base16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr))) {}

  friend is_utf8_really_inline Mask operator==(const simd16<T> lhs,
                                               const simd16<T> rhs) {
    return _mm_cmpeq_epi16(lhs, rhs);
  }

  static const int SIZE = sizeof(base<simd16<T>>::value);

  template <int N = 1>
  is_utf8_really_inline simd16<T> prev(const simd16<T> prev_chunk) const {
    return _mm_alignr_epi8(*this, prev_chunk, 16 - N);
  }
};

// SIMD byte mask type (returned by things like eq and gt)
template <> struct simd16<bool> : base16<bool> {
  static is_utf8_really_inline simd16<bool> splat(bool _value) {
    return _mm_set1_epi16(uint16_t(-(!!_value)));
  }

  is_utf8_really_inline simd16<bool>() : base16() {}
  is_utf8_really_inline simd16<bool>(const __m128i _value)
      : base16<bool>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd16<bool>(bool _value)
      : base16<bool>(splat(_value)) {}

  is_utf8_really_inline int to_bitmask() const {
    return _mm_movemask_epi8(*this);
  }
  is_utf8_really_inline bool any() const {
    return !_mm_testz_si128(*this, *this);
  }
  is_utf8_really_inline simd16<bool> operator~() const { return *this ^ true; }
};

template <typename T> struct base16_numeric : base16<T> {
  static is_utf8_really_inline simd16<T> splat(T _value) {
    return _mm_set1_epi16(_value);
  }
  static is_utf8_really_inline simd16<T> zero() { return _mm_setzero_si128(); }
  static is_utf8_really_inline simd16<T> load(const T values[8]) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(values));
  }

  is_utf8_really_inline base16_numeric() : base16<T>() {}
  is_utf8_really_inline base16_numeric(const __m128i _value)
      : base16<T>(_value) {}

  // Store to array
  is_utf8_really_inline void store(T dst[8]) const {
    return _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), *this);
  }

  // Override to distinguish from bool version
  is_utf8_really_inline simd16<T> operator~() const { return *this ^ 0xFFu; }

  // Addition/subtraction are the same for signed and unsigned
  is_utf8_really_inline simd16<T> operator+(const simd16<T> other) const {
    return _mm_add_epi16(*this, other);
  }
  is_utf8_really_inline simd16<T> operator-(const simd16<T> other) const {
    return _mm_sub_epi16(*this, other);
  }
  is_utf8_really_inline simd16<T> &operator+=(const simd16<T> other) {
    *this = *this + other;
    return *static_cast<simd16<T> *>(this);
  }
  is_utf8_really_inline simd16<T> &operator-=(const simd16<T> other) {
    *this = *this - other;
    return *static_cast<simd16<T> *>(this);
  }
};

// Signed words
template <> struct simd16<int16_t> : base16_numeric<int16_t> {
  is_utf8_really_inline simd16() : base16_numeric<int16_t>() {}
  is_utf8_really_inline simd16(const __m128i _value)
      : base16_numeric<int16_t>(_value) {}
  // Splat constructor
  is_utf8_really_inline simd16(int16_t _value) : simd16(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd16(const int16_t *values) : simd16(load(values)) {}
  is_utf8_really_inline simd16(const char16_t *values)
      : simd16(load(reinterpret_cast<const int16_t *>(values))) {}
  // Member-by-member initialization
  is_utf8_really_inline simd16(int16_t v0, int16_t v1, int16_t v2, int16_t v3,
                               int16_t v4, int16_t v5, int16_t v6, int16_t v7)
      : simd16(_mm_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7)) {}
  is_utf8_really_inline operator simd16<uint16_t>() const;

  // Order-sensitive comparisons
  is_utf8_really_inline simd16<int16_t>
  max_val(const simd16<int16_t> other) const {
    return _mm_max_epi16(*this, other);
  }
  is_utf8_really_inline simd16<int16_t>
  min_val(const simd16<int16_t> other) const {
    return _mm_min_epi16(*this, other);
  }
  is_utf8_really_inline simd16<bool>
  operator>(const simd16<int16_t> other) const {
    return _mm_cmpgt_epi16(*this, other);
  }
  is_utf8_really_inline simd16<bool>
  operator<(const simd16<int16_t> other) const {
    return _mm_cmpgt_epi16(other, *this);
  }
};

// Unsigned words
template <> struct simd16<uint16_t> : base16_numeric<uint16_t> {
  is_utf8_really_inline simd16() : base16_numeric<uint16_t>() {}
  is_utf8_really_inline simd16(const __m128i _value)
      : base16_numeric<uint16_t>(_value) {}

  // Splat constructor
  is_utf8_really_inline simd16(uint16_t _value) : simd16(splat(_value)) {}
  // Array constructor
  is_utf8_really_inline simd16(const uint16_t *values) : simd16(load(values)) {}
  is_utf8_really_inline simd16(const char16_t *values)
      : simd16(load(reinterpret_cast<const uint16_t *>(values))) {}
  // Member-by-member initialization
  is_utf8_really_inline simd16(uint16_t v0, uint16_t v1, uint16_t v2,
                               uint16_t v3, uint16_t v4, uint16_t v5,
                               uint16_t v6, uint16_t v7)
      : simd16(_mm_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7)) {}
  // Repeat 16 values as many times as necessary (usually for lookup tables)
  is_utf8_really_inline static simd16<uint16_t>
  repeat_16(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3, uint16_t v4,
            uint16_t v5, uint16_t v6, uint16_t v7) {
    return simd16<uint16_t>(v0, v1, v2, v3, v4, v5, v6, v7);
  }

  // Saturated math
  is_utf8_really_inline simd16<uint16_t>
  saturating_add(const simd16<uint16_t> other) const {
    return _mm_adds_epu16(*this, other);
  }
  is_utf8_really_inline simd16<uint16_t>
  saturating_sub(const simd16<uint16_t> other) const {
    return _mm_subs_epu16(*this, other);
  }

  // Order-specific operations
  is_utf8_really_inline simd16<uint16_t>
  max_val(const simd16<uint16_t> other) const {
    return _mm_max_epu16(*this, other);
  }
  is_utf8_really_inline simd16<uint16_t>
  min_val(const simd16<uint16_t> other) const {
    return _mm_min_epu16(*this, other);
  }
  // Same as >, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd16<uint16_t>
  gt_bits(const simd16<uint16_t> other) const {
    return this->saturating_sub(other);
  }
  // Same as <, but only guarantees true is nonzero (< guarantees true = -1)
  is_utf8_really_inline simd16<uint16_t>
  lt_bits(const simd16<uint16_t> other) const {
    return other.saturating_sub(*this);
  }
  is_utf8_really_inline simd16<bool>
  operator<=(const simd16<uint16_t> other) const {
    return other.max_val(*this) == other;
  }
  is_utf8_really_inline simd16<bool>
  operator>=(const simd16<uint16_t> other) const {
    return other.min_val(*this) == other;
  }
  is_utf8_really_inline simd16<bool>
  operator>(const simd16<uint16_t> other) const {
    return this->gt_bits(other).any_bits_set();
  }
  is_utf8_really_inline simd16<bool>
  operator<(const simd16<uint16_t> other) const {
    return this->gt_bits(other).any_bits_set();
  }

  // Bit-specific operations
  is_utf8_really_inline simd16<bool> bits_not_set() const {
    return *this == uint16_t(0);
  }
  is_utf8_really_inline simd16<bool> bits_not_set(simd16<uint16_t> bits) const {
    return (*this & bits).bits_not_set();
  }
  is_utf8_really_inline simd16<bool> any_bits_set() const {
    return ~this->bits_not_set();
  }
  is_utf8_really_inline simd16<bool> any_bits_set(simd16<uint16_t> bits) const {
    return ~this->bits_not_set(bits);
  }

  is_utf8_really_inline bool bits_not_set_anywhere() const {
    return _mm_testz_si128(*this, *this);
  }
  is_utf8_really_inline bool any_bits_set_anywhere() const {
    return !bits_not_set_anywhere();
  }
  is_utf8_really_inline bool
  bits_not_set_anywhere(simd16<uint16_t> bits) const {
    return _mm_testz_si128(*this, bits);
  }
  is_utf8_really_inline bool
  any_bits_set_anywhere(simd16<uint16_t> bits) const {
    return !bits_not_set_anywhere(bits);
  }
  template <int N> is_utf8_really_inline simd16<uint16_t> shr() const {
    return simd16<uint16_t>(_mm_srli_epi16(*this, N));
  }
  template <int N> is_utf8_really_inline simd16<uint16_t> shl() const {
    return simd16<uint16_t>(_mm_slli_epi16(*this, N));
  }
  // Get one of the bits and make a bitmask out of it.
  // e.g. value.get_bit<7>() gets the high bit
  template <int N> is_utf8_really_inline int get_bit() const {
    return _mm_movemask_epi8(_mm_slli_epi16(*this, 7 - N));
  }

  // Change the endianness
  is_utf8_really_inline simd16<uint16_t> swap_bytes() const {
    const __m128i swap =
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    return _mm_shuffle_epi8(*this, swap);
  }

  // Pack with the unsigned saturation  two uint16_t words into single uint8_t
  // vector
  static is_utf8_really_inline simd8<uint8_t> pack(const simd16<uint16_t> &v0,
                                                   const simd16<uint16_t> &v1) {
    return _mm_packus_epi16(v0, v1);
  }
};
is_utf8_really_inline simd16<int16_t>::operator simd16<uint16_t>() const {
  return this->value;
}

template <typename T> struct simd16x32 {
  static constexpr int NUM_CHUNKS = 64 / sizeof(simd16<T>);
  static_assert(NUM_CHUNKS == 4,
                "Westmere kernel should use four registers per 64-byte block.");
  simd16<T> chunks[NUM_CHUNKS];

  simd16x32(const simd16x32<T> &o) = delete; // no copy allowed
  simd16x32<T> &
  operator=(const simd16<T> other) = delete; // no assignment allowed
  simd16x32() = delete;                      // no default constructor allowed

  is_utf8_really_inline
  simd16x32(const simd16<T> chunk0, const simd16<T> chunk1,
            const simd16<T> chunk2, const simd16<T> chunk3)
      : chunks{chunk0, chunk1, chunk2, chunk3} {}
  is_utf8_really_inline simd16x32(const T *ptr)
      : chunks{simd16<T>::load(ptr),
               simd16<T>::load(ptr + sizeof(simd16<T>) / sizeof(T)),
               simd16<T>::load(ptr + 2 * sizeof(simd16<T>) / sizeof(T)),
               simd16<T>::load(ptr + 3 * sizeof(simd16<T>) / sizeof(T))} {}

  is_utf8_really_inline void store(T *ptr) const {
    this->chunks[0].store(ptr + sizeof(simd16<T>) * 0 / sizeof(T));
    this->chunks[1].store(ptr + sizeof(simd16<T>) * 1 / sizeof(T));
    this->chunks[2].store(ptr + sizeof(simd16<T>) * 2 / sizeof(T));
    this->chunks[3].store(ptr + sizeof(simd16<T>) * 3 / sizeof(T));
  }

  is_utf8_really_inline simd16<T> reduce_or() const {
    return (this->chunks[0] | this->chunks[1]) |
           (this->chunks[2] | this->chunks[3]);
  }

  is_utf8_really_inline bool is_ascii() const {
    return this->reduce_or().is_ascii();
  }

  is_utf8_really_inline void store_ascii_as_utf16(char16_t *ptr) const {
    this->chunks[0].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 0);
    this->chunks[1].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 1);
    this->chunks[2].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 2);
    this->chunks[3].store_ascii_as_utf16(ptr + sizeof(simd16<T>) * 3);
  }

  is_utf8_really_inline uint64_t to_bitmask() const {
    uint64_t r0 = uint32_t(this->chunks[0].to_bitmask());
    uint64_t r1 = this->chunks[1].to_bitmask();
    uint64_t r2 = this->chunks[2].to_bitmask();
    uint64_t r3 = this->chunks[3].to_bitmask();
    return r0 | (r1 << 16) | (r2 << 32) | (r3 << 48);
  }

  is_utf8_really_inline void swap_bytes() {
    this->chunks[0] = this->chunks[0].swap_bytes();
    this->chunks[1] = this->chunks[1].swap_bytes();
    this->chunks[2] = this->chunks[2].swap_bytes();
    this->chunks[3] = this->chunks[3].swap_bytes();
  }

  is_utf8_really_inline uint64_t eq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] == mask, this->chunks[1] == mask,
                           this->chunks[2] == mask, this->chunks[3] == mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t eq(const simd16x32<uint16_t> &other) const {
    return simd16x32<bool>(this->chunks[0] == other.chunks[0],
                           this->chunks[1] == other.chunks[1],
                           this->chunks[2] == other.chunks[2],
                           this->chunks[3] == other.chunks[3])
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t lteq(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] <= mask, this->chunks[1] <= mask,
                           this->chunks[2] <= mask, this->chunks[3] <= mask)
        .to_bitmask();
  }

  is_utf8_really_inline uint64_t in_range(const T low, const T high) const {
    const simd16<T> mask_low = simd16<T>::splat(low);
    const simd16<T> mask_high = simd16<T>::splat(high);

    return simd16x32<bool>(
               (this->chunks[0] <= mask_high) & (this->chunks[0] >= mask_low),
               (this->chunks[1] <= mask_high) & (this->chunks[1] >= mask_low),
               (this->chunks[2] <= mask_high) & (this->chunks[2] >= mask_low),
               (this->chunks[3] <= mask_high) & (this->chunks[3] >= mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t not_in_range(const T low, const T high) const {
    const simd16<T> mask_low = simd16<T>::splat(static_cast<T>(low - 1));
    const simd16<T> mask_high = simd16<T>::splat(static_cast<T>(high + 1));
    return simd16x32<bool>(
               (this->chunks[0] >= mask_high) | (this->chunks[0] <= mask_low),
               (this->chunks[1] >= mask_high) | (this->chunks[1] <= mask_low),
               (this->chunks[2] >= mask_high) | (this->chunks[2] <= mask_low),
               (this->chunks[3] >= mask_high) | (this->chunks[3] <= mask_low))
        .to_bitmask();
  }
  is_utf8_really_inline uint64_t lt(const T m) const {
    const simd16<T> mask = simd16<T>::splat(m);
    return simd16x32<bool>(this->chunks[0] < mask, this->chunks[1] < mask,
                           this->chunks[2] < mask, this->chunks[3] < mask)
        .to_bitmask();
  }
}; // struct simd16x32<T>

} // namespace simd
} // unnamed namespace
} // namespace westmere
} // namespace is_utf8_internals

#endif // IS_UTF8_WESTMERE_SIMD_INPUT_H

IS_UTF8_UNTARGET_REGION

#endif // IS_UTF8_IMPLEMENTATION_WESTMERE
#endif // IS_UTF8_WESTMERE_COMMON_H

#ifndef IS_UTF8_FALLBACK_H
#define IS_UTF8_FALLBACK_H

// Note that fallback.h is always imported last.

// Default Fallback to on unless a builtin implementation has already been
// selected.
#ifndef IS_UTF8_IMPLEMENTATION_FALLBACK
#if IS_UTF8_CAN_ALWAYS_RUN_ARM64 || IS_UTF8_CAN_ALWAYS_RUN_ICELAKE ||          \
    IS_UTF8_CAN_ALWAYS_RUN_HASWELL || IS_UTF8_CAN_ALWAYS_RUN_WESTMERE ||       \
    IS_UTF8_CAN_ALWAYS_RUN_PPC64
#define IS_UTF8_IMPLEMENTATION_FALLBACK 0
#else
#define IS_UTF8_IMPLEMENTATION_FALLBACK 1
#endif
#endif

#if IS_UTF8_IMPLEMENTATION_FALLBACK
#define IS_UTF8_CAN_ALWAYS_RUN_FALLBACK 1
#else
#define IS_UTF8_CAN_ALWAYS_RUN_FALLBACK 0
#endif

#if IS_UTF8_IMPLEMENTATION_FALLBACK

namespace is_utf8_internals {
/**
 * Fallback implementation (runs on any machine).
 */
namespace fallback {} // namespace fallback
} // namespace is_utf8_internals

#ifndef IS_UTF8_FALLBACK_IMPLEMENTATION_H
#define IS_UTF8_FALLBACK_IMPLEMENTATION_H

namespace is_utf8_internals {
namespace fallback {

class implementation final : public is_utf8_internals::implementation {
public:
  is_utf8_really_inline implementation()
      : is_utf8_internals::implementation(
            "fallback", "Generic fallback implementation", 0) {}
  is_utf8_warn_unused bool validate_utf8(const char *buf,
                                         size_t len) const noexcept final;
};

} // namespace fallback
} // namespace is_utf8_internals

#endif // IS_UTF8_FALLBACK_IMPLEMENTATION_H

// redefining IS_UTF8_IMPLEMENTATION to "fallback"
// #define IS_UTF8_IMPLEMENTATION fallback

// Declarations

#ifndef IS_UTF8_FALLBACK_BITMANIPULATION_H
#define IS_UTF8_FALLBACK_BITMANIPULATION_H

#include <limits>

namespace is_utf8_internals {
namespace fallback {
namespace {

#if defined(_MSC_VER) && !defined(_M_ARM64) && !defined(_M_X64)
static inline unsigned char _BitScanForward64(unsigned long *ret, uint64_t x) {
  unsigned long x0 = (unsigned long)x, top, bottom;
  _BitScanForward(&top, (unsigned long)(x >> 32));
  _BitScanForward(&bottom, x0);
  *ret = x0 ? bottom : 32 + top;
  return x != 0;
}
static unsigned char _BitScanReverse64(unsigned long *ret, uint64_t x) {
  unsigned long x1 = (unsigned long)(x >> 32), top, bottom;
  _BitScanReverse(&top, x1);
  _BitScanReverse(&bottom, (unsigned long)x);
  *ret = x1 ? top + 32 : bottom;
  return x != 0;
}
#endif

} // unnamed namespace
} // namespace fallback
} // namespace is_utf8_internals

#endif // IS_UTF8_FALLBACK_BITMANIPULATION_H

#endif // IS_UTF8_IMPLEMENTATION_FALLBACK
#endif // IS_UTF8_FALLBACK_H

namespace is_utf8_internals {
bool implementation::supported_by_runtime_system() const {
  uint32_t required_instruction_sets = this->required_instruction_sets();
  uint32_t supported_instruction_sets =
      internal::detect_supported_architectures();
  return ((supported_instruction_sets & required_instruction_sets) ==
          required_instruction_sets);
}

namespace internal {

// Static array of known implementations. We're hoping these get baked into the
// executable without requiring a static initializer.

#if IS_UTF8_IMPLEMENTATION_ICELAKE
const icelake::implementation icelake_singleton{};
#endif
#if IS_UTF8_IMPLEMENTATION_HASWELL
const haswell::implementation haswell_singleton{};
#endif
#if IS_UTF8_IMPLEMENTATION_WESTMERE
const westmere::implementation westmere_singleton{};
#endif
#if IS_UTF8_IMPLEMENTATION_ARM64
const arm64::implementation arm64_singleton{};
#endif
#if IS_UTF8_IMPLEMENTATION_PPC64
const ppc64::implementation ppc64_singleton{};
#endif
#if IS_UTF8_IMPLEMENTATION_FALLBACK
const fallback::implementation fallback_singleton{};
#endif

/**
 * @private Detects best supported implementation on first use, and sets it
 */
class detect_best_supported_implementation_on_first_use final
    : public implementation {
public:
  const std::string &name() const noexcept final { return set_best()->name(); }
  const std::string &description() const noexcept final {
    return set_best()->description();
  }
  uint32_t required_instruction_sets() const noexcept final {
    return set_best()->required_instruction_sets();
  }

  is_utf8_warn_unused bool
  validate_utf8(const char *buf, size_t len) const noexcept final override {
    return set_best()->validate_utf8(buf, len);
  }

  is_utf8_really_inline
  detect_best_supported_implementation_on_first_use() noexcept
      : implementation("best_supported_detector",
                       "Detects the best supported implementation and sets it",
                       0) {}

private:
  const implementation *set_best() const noexcept;
};

const detect_best_supported_implementation_on_first_use
    detect_best_supported_implementation_on_first_use_singleton;

const std::initializer_list<const implementation *>
    available_implementation_pointers {
#if IS_UTF8_IMPLEMENTATION_ICELAKE
  &icelake_singleton,
#endif
#if IS_UTF8_IMPLEMENTATION_HASWELL
      &haswell_singleton,
#endif
#if IS_UTF8_IMPLEMENTATION_WESTMERE
      &westmere_singleton,
#endif
#if IS_UTF8_IMPLEMENTATION_ARM64
      &arm64_singleton,
#endif
#if IS_UTF8_IMPLEMENTATION_PPC64
      &ppc64_singleton,
#endif
#if IS_UTF8_IMPLEMENTATION_FALLBACK
      &fallback_singleton,
#endif
}; // available_implementation_pointers

// So we can return UNSUPPORTED_ARCHITECTURE from the parser when there is no
// support
class unsupported_implementation final : public implementation {
public:
  is_utf8_warn_unused bool validate_utf8(const char *,
                                         size_t) const noexcept final override {
    return false; // Just refuse to validate. Given that we have a fallback
                  // implementation
    // it seems unlikely that unsupported_implementation will ever be used. If
    // it is used, then it will flag all strings as invalid. The alternative is
    // to return an error_code from which the user has to figure out whether the
    // string is valid UTF-8... which seems like a lot of work just to handle
    // the very unlikely case that we have an unsupported implementation. And,
    // when it does happen (that we have an unsupported implementation), what
    // are the chances that the programmer has a fallback? Given that *we*
    // provide the fallback, it implies that the programmer would need a
    // fallback for our fallback.
  }

  unsupported_implementation()
      : implementation("unsupported",
                       "Unsupported CPU (no detected SIMD instructions)", 0) {}
};

const unsupported_implementation unsupported_singleton{};

size_t available_implementation_list::size() const noexcept {
  return internal::available_implementation_pointers.size();
}
const implementation *const *
available_implementation_list::begin() const noexcept {
  return internal::available_implementation_pointers.begin();
}
const implementation *const *
available_implementation_list::end() const noexcept {
  return internal::available_implementation_pointers.end();
}
const implementation *
available_implementation_list::detect_best_supported() const noexcept {
  // They are prelisted in priority order, so we just go down the list
  uint32_t supported_instruction_sets =
      internal::detect_supported_architectures();
  for (const implementation *impl :
       internal::available_implementation_pointers) {
    uint32_t required_instruction_sets = impl->required_instruction_sets();
    if ((supported_instruction_sets & required_instruction_sets) ==
        required_instruction_sets) {
      return impl;
    }
  }
  return &unsupported_singleton; // this should never happen?
}

const implementation *
detect_best_supported_implementation_on_first_use::set_best() const noexcept {
  IS_UTF8_PUSH_DISABLE_WARNINGS
  IS_UTF8_DISABLE_DEPRECATED_WARNING // Disable CRT_SECURE warning on MSVC:
                                     // manually verified this is safe
      char *force_implementation_name = getenv("IS_UTF8_FORCE_IMPLEMENTATION");
  IS_UTF8_POP_DISABLE_WARNINGS

  if (force_implementation_name) {
    auto force_implementation =
        get_available_implementations()[force_implementation_name];
    if (force_implementation) {
      return get_active_implementation() = force_implementation;
    } else {
      // Note: abort() and stderr usage within the library is forbidden.
      return get_active_implementation() = &unsupported_singleton;
    }
  }
  return get_active_implementation() =
             get_available_implementations().detect_best_supported();
}

} // namespace internal

IS_UTF8_DLLIMPORTEXPORT const internal::available_implementation_list& get_available_implementations() {
  static const internal::available_implementation_list available_implementations{};
  return available_implementations;
}

IS_UTF8_DLLIMPORTEXPORT internal::atomic_ptr<const implementation>& get_active_implementation() {
    static const internal::detect_best_supported_implementation_on_first_use detect_best_supported_implementation_on_first_use_singleton;
    static internal::atomic_ptr<const implementation> active_implementation{&detect_best_supported_implementation_on_first_use_singleton};
    return active_implementation;
}


is_utf8_warn_unused bool validate_utf8(const char *buf, size_t len) noexcept {
  return get_active_implementation()->validate_utf8(buf, len);
}

const implementation *builtin_implementation() {
  static const implementation *builtin_impl =
      get_available_implementations()[IS_UTF8_STRINGIFY(
          IS_UTF8_BUILTIN_IMPLEMENTATION)];
  return builtin_impl;
}

} // namespace is_utf8_internals

// The scalar routines should be included once.

#ifndef IS_UTF8_UTF8_H
#define IS_UTF8_UTF8_H

namespace is_utf8_internals {
namespace scalar {
namespace {
namespace utf8 {
#if IS_UTF8_IS_ARM64
// not needed
#else
// credit: based on code from Google Fuchsia (Apache Licensed)
inline is_utf8_warn_unused bool validate(const char *buf, size_t len) noexcept {
  const uint8_t *data = reinterpret_cast<const uint8_t *>(buf);
  uint64_t pos = 0;
  uint32_t code_point = 0;
  while (pos < len) {
    // check of the next 8 bytes are ascii.
    uint64_t next_pos = pos + 16;
    if (next_pos <=
        len) { // if it is safe to read 8 more bytes, check that they are ascii
      uint64_t v1;
      std::memcpy(&v1, data + pos, sizeof(uint64_t));
      uint64_t v2;
      std::memcpy(&v2, data + pos + sizeof(uint64_t), sizeof(uint64_t));
      uint64_t v{v1 | v2};
      if ((v & 0x8080808080808080) == 0) {
        pos = next_pos;
        continue;
      }
    }
    unsigned char byte = data[pos];

    while (byte < 0b10000000) {
      if (++pos == len) {
        return true;
      }
      byte = data[pos];
    }

    if ((byte & 0b11100000) == 0b11000000) {
      next_pos = pos + 2;
      if (next_pos > len) {
        return false;
      }
      if ((data[pos + 1] & 0b11000000) != 0b10000000) {
        return false;
      }
      // range check
      code_point = (byte & 0b00011111) << 6 | (data[pos + 1] & 0b00111111);
      if ((code_point < 0x80) || (0x7ff < code_point)) {
        return false;
      }
    } else if ((byte & 0b11110000) == 0b11100000) {
      next_pos = pos + 3;
      if (next_pos > len) {
        return false;
      }
      if ((data[pos + 1] & 0b11000000) != 0b10000000) {
        return false;
      }
      if ((data[pos + 2] & 0b11000000) != 0b10000000) {
        return false;
      }
      // range check
      code_point = (byte & 0b00001111) << 12 |
                   (data[pos + 1] & 0b00111111) << 6 |
                   (data[pos + 2] & 0b00111111);
      if ((code_point < 0x800) || (0xffff < code_point) ||
          (0xd7ff < code_point && code_point < 0xe000)) {
        return false;
      }
    } else if ((byte & 0b11111000) == 0b11110000) { // 0b11110000
      next_pos = pos + 4;
      if (next_pos > len) {
        return false;
      }
      if ((data[pos + 1] & 0b11000000) != 0b10000000) {
        return false;
      }
      if ((data[pos + 2] & 0b11000000) != 0b10000000) {
        return false;
      }
      if ((data[pos + 3] & 0b11000000) != 0b10000000) {
        return false;
      }
      // range check
      code_point =
          (byte & 0b00000111) << 18 | (data[pos + 1] & 0b00111111) << 12 |
          (data[pos + 2] & 0b00111111) << 6 | (data[pos + 3] & 0b00111111);
      if (code_point <= 0xffff || 0x10ffff < code_point) {
        return false;
      }
    } else {
      // we may have a continuation
      return false;
    }
    pos = next_pos;
  }
  return true;
}
#endif
} // namespace utf8
} // unnamed namespace
} // namespace scalar
} // namespace is_utf8_internals

#endif

IS_UTF8_PUSH_DISABLE_WARNINGS
IS_UTF8_DISABLE_UNDESIRED_WARNINGS

#if IS_UTF8_IMPLEMENTATION_ARM64

// redefining IS_UTF8_IMPLEMENTATION to "arm64"
// #define IS_UTF8_IMPLEMENTATION arm64

namespace is_utf8_internals {
namespace arm64 {
namespace {
#ifndef IS_UTF8_ARM64_H
#error "arm64.h must be included"
#endif
using namespace simd;

is_utf8_really_inline bool is_ascii(const simd8x64<uint8_t> &input) {
  simd8<uint8_t> bits = input.reduce_or();
  return bits.max_val() < 0b10000000u;
}

is_utf8_unused is_utf8_really_inline simd8<bool>
must_be_continuation(const simd8<uint8_t> prev1, const simd8<uint8_t> prev2,
                     const simd8<uint8_t> prev3) {
  simd8<bool> is_second_byte = prev1 >= uint8_t(0b11000000u);
  simd8<bool> is_third_byte = prev2 >= uint8_t(0b11100000u);
  simd8<bool> is_fourth_byte = prev3 >= uint8_t(0b11110000u);
  // Use ^ instead of | for is_*_byte, because ^ is commutative, and the caller
  // is using ^ as well. This will work fine because we only have to report
  // errors for cases with 0-1 lead bytes. Multiple lead bytes implies 2
  // overlapping multibyte characters, and if that happens, there is guaranteed
  // to be at least *one* lead byte that is part of only 1 other multibyte
  // character. The error will be detected there.
  return is_second_byte ^ is_third_byte ^ is_fourth_byte;
}

is_utf8_really_inline simd8<bool>
must_be_2_3_continuation(const simd8<uint8_t> prev2,
                         const simd8<uint8_t> prev3) {
  simd8<bool> is_third_byte = prev2 >= uint8_t(0b11100000u);
  simd8<bool> is_fourth_byte = prev3 >= uint8_t(0b11110000u);
  return is_third_byte ^ is_fourth_byte;
}

} // unnamed namespace
} // namespace arm64
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace arm64 {
namespace {

// Walks through a buffer in block-sized increments, loading the last part with
// spaces
template <size_t STEP_SIZE> struct buf_block_reader {
public:
  is_utf8_really_inline buf_block_reader(const uint8_t *_buf, size_t _len);
  is_utf8_really_inline size_t block_index();
  is_utf8_really_inline bool has_full_block() const;
  is_utf8_really_inline const uint8_t *full_block() const;
  /**
   * Get the last block, padded with spaces.
   *
   * There will always be a last block, with at least 1 byte, unless len == 0
   * (in which case this function fills the buffer with spaces and returns 0. In
   * particular, if len == STEP_SIZE there will be 0 full_blocks and 1 remainder
   * block with STEP_SIZE bytes and no spaces for padding.
   *
   * @return the number of effective characters in the last block.
   */
  is_utf8_really_inline size_t get_remainder(uint8_t *dst) const;
  is_utf8_really_inline void advance();

private:
  const uint8_t *buf;
  const size_t len;
  const size_t lenminusstep;
  size_t idx;
};

// Routines to print masks and text for debugging bitmask operations
is_utf8_unused static char *format_input_text_64(const uint8_t *text) {
  static char *buf =
      reinterpret_cast<char *>(malloc(sizeof(simd8x64<uint8_t>) + 1));
  for (size_t i = 0; i < sizeof(simd8x64<uint8_t>); i++) {
    buf[i] = int8_t(text[i]) < ' ' ? '_' : int8_t(text[i]);
  }
  buf[sizeof(simd8x64<uint8_t>)] = '\0';
  return buf;
}

// Routines to print masks and text for debugging bitmask operations
is_utf8_unused static char *format_input_text(const simd8x64<uint8_t> &in) {
  static char *buf =
      reinterpret_cast<char *>(malloc(sizeof(simd8x64<uint8_t>) + 1));
  in.store(reinterpret_cast<uint8_t *>(buf));
  for (size_t i = 0; i < sizeof(simd8x64<uint8_t>); i++) {
    if (buf[i] < ' ') {
      buf[i] = '_';
    }
  }
  buf[sizeof(simd8x64<uint8_t>)] = '\0';
  return buf;
}

is_utf8_unused static char *format_mask(uint64_t mask) {
  static char *buf = reinterpret_cast<char *>(malloc(64 + 1));
  for (size_t i = 0; i < 64; i++) {
    buf[i] = (mask & (size_t(1) << i)) ? 'X' : ' ';
  }
  buf[64] = '\0';
  return buf;
}

template <size_t STEP_SIZE>
is_utf8_really_inline
buf_block_reader<STEP_SIZE>::buf_block_reader(const uint8_t *_buf, size_t _len)
    : buf{_buf}, len{_len},
      lenminusstep{len < STEP_SIZE ? 0 : len - STEP_SIZE}, idx{0} {}

template <size_t STEP_SIZE>
is_utf8_really_inline size_t buf_block_reader<STEP_SIZE>::block_index() {
  return idx;
}

template <size_t STEP_SIZE>
is_utf8_really_inline bool buf_block_reader<STEP_SIZE>::has_full_block() const {
  return idx < lenminusstep;
}

template <size_t STEP_SIZE>
is_utf8_really_inline const uint8_t *
buf_block_reader<STEP_SIZE>::full_block() const {
  return &buf[idx];
}

template <size_t STEP_SIZE>
is_utf8_really_inline size_t
buf_block_reader<STEP_SIZE>::get_remainder(uint8_t *dst) const {
  if (len == idx) {
    return 0;
  } // memcpy(dst, null, 0) will trigger an error with some sanitizers
  std::memset(dst, 0x20,
              STEP_SIZE); // std::memset STEP_SIZE because it's more efficient
                          // to write out 8 or 16 bytes at once.
  std::memcpy(dst, buf + idx, len - idx);
  return len - idx;
}

template <size_t STEP_SIZE>
is_utf8_really_inline void buf_block_reader<STEP_SIZE>::advance() {
  idx += STEP_SIZE;
}

} // unnamed namespace
} // namespace arm64
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace arm64 {
namespace {
namespace utf8_validation {

using namespace simd;

is_utf8_really_inline simd8<uint8_t>
check_special_cases(const simd8<uint8_t> input, const simd8<uint8_t> prev1) {
  // Bit 0 = Too Short (lead byte/ASCII followed by lead byte/ASCII)
  // Bit 1 = Too Long (ASCII followed by continuation)
  // Bit 2 = Overlong 3-byte
  // Bit 4 = Surrogate
  // Bit 5 = Overlong 2-byte
  // Bit 7 = Two Continuations
  constexpr const uint8_t TOO_SHORT = 1 << 0;  // 11______ 0_______
                                               // 11______ 11______
  constexpr const uint8_t TOO_LONG = 1 << 1;   // 0_______ 10______
  constexpr const uint8_t OVERLONG_3 = 1 << 2; // 11100000 100_____
  constexpr const uint8_t SURROGATE = 1 << 4;  // 11101101 101_____
  constexpr const uint8_t OVERLONG_2 = 1 << 5; // 1100000_ 10______
  constexpr const uint8_t TWO_CONTS = 1 << 7;  // 10______ 10______
  constexpr const uint8_t TOO_LARGE = 1 << 3;  // 11110100 1001____
                                               // 11110100 101_____
                                               // 11110101 1001____
                                               // 11110101 101_____
                                               // 1111011_ 1001____
                                               // 1111011_ 101_____
                                               // 11111___ 1001____
                                               // 11111___ 101_____
  constexpr const uint8_t TOO_LARGE_1000 = 1 << 6;
  // 11110101 1000____
  // 1111011_ 1000____
  // 11111___ 1000____
  constexpr const uint8_t OVERLONG_4 = 1 << 6; // 11110000 1000____

  const simd8<uint8_t> byte_1_high = prev1.shr<4>().lookup_16<uint8_t>(
      // 0_______ ________ <ASCII in byte 1>
      TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
      TOO_LONG,
      // 10______ ________ <continuation in byte 1>
      TWO_CONTS, TWO_CONTS, TWO_CONTS, TWO_CONTS,
      // 1100____ ________ <two byte lead in byte 1>
      TOO_SHORT | OVERLONG_2,
      // 1101____ ________ <two byte lead in byte 1>
      TOO_SHORT,
      // 1110____ ________ <three byte lead in byte 1>
      TOO_SHORT | OVERLONG_3 | SURROGATE,
      // 1111____ ________ <four+ byte lead in byte 1>
      TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4);
  constexpr const uint8_t CARRY =
      TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 .
  const simd8<uint8_t> byte_1_low =
      (prev1 & 0x0F)
          .lookup_16<uint8_t>(
              // ____0000 ________
              CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
              // ____0001 ________
              CARRY | OVERLONG_2,
              // ____001_ ________
              CARRY, CARRY,

              // ____0100 ________
              CARRY | TOO_LARGE,
              // ____0101 ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              // ____011_ ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,

              // ____1___ ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              // ____1101 ________
              CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000);
  const simd8<uint8_t> byte_2_high = input.shr<4>().lookup_16<uint8_t>(
      // ________ 0_______ <ASCII in byte 2>
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
      TOO_SHORT, TOO_SHORT,

      // ________ 1000____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 |
          OVERLONG_4,
      // ________ 1001____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
      // ________ 101_____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,

      // ________ 11______
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT);
  return (byte_1_high & byte_1_low & byte_2_high);
}
is_utf8_really_inline simd8<uint8_t>
check_multibyte_lengths(const simd8<uint8_t> input,
                        const simd8<uint8_t> prev_input,
                        const simd8<uint8_t> sc) {
  simd8<uint8_t> prev2 = input.prev<2>(prev_input);
  simd8<uint8_t> prev3 = input.prev<3>(prev_input);
  simd8<uint8_t> must23 =
      simd8<uint8_t>(must_be_2_3_continuation(prev2, prev3));
  simd8<uint8_t> must23_80 = must23 & uint8_t(0x80);
  return must23_80 ^ sc;
}

//
// Return nonzero if there are incomplete multibyte characters at the end of the
// block: e.g. if there is a 4-byte character, but it's 3 bytes from the end.
//
is_utf8_really_inline simd8<uint8_t> is_incomplete(const simd8<uint8_t> input) {
  // If the previous input's last 3 bytes match this, they're too short (they
  // ended at EOF):
  // ... 1111____ 111_____ 11______
  static const uint8_t max_array[32] = {255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        0b11110000u - 1,
                                        0b11100000u - 1,
                                        0b11000000u - 1};
  const simd8<uint8_t> max_value(
      &max_array[sizeof(max_array) - sizeof(simd8<uint8_t>)]);
  return input.gt_bits(max_value);
}

struct utf8_checker {
  // If this is nonzero, there has been a UTF-8 error.
  simd8<uint8_t> error;
  // The last input we received
  simd8<uint8_t> prev_input_block;
  // Whether the last input we received was incomplete (used for ASCII fast
  // path)
  simd8<uint8_t> prev_incomplete;

  //
  // Check whether the current bytes are valid UTF-8.
  //
  is_utf8_really_inline void check_utf8_bytes(const simd8<uint8_t> input,
                                              const simd8<uint8_t> prev_input) {
    // Flip prev1...prev3 so we can easily determine if they are 2+, 3+ or 4+
    // lead bytes (2, 3, 4-byte leads become large positive numbers instead of
    // small negative numbers)
    simd8<uint8_t> prev1 = input.prev<1>(prev_input);
    simd8<uint8_t> sc = check_special_cases(input, prev1);
    this->error |= check_multibyte_lengths(input, prev_input, sc);
  }

  // The only problem that can happen at EOF is that a multibyte character is
  // too short or a byte value too large in the last bytes: check_special_cases
  // only checks for bytes too large in the first of two bytes.
  is_utf8_really_inline void check_eof() {
    // If the previous block had incomplete UTF-8 characters at the end, an
    // ASCII block can't possibly finish them.
    this->error |= this->prev_incomplete;
  }

  is_utf8_really_inline void check_next_input(const simd8x64<uint8_t> &input) {
    if (is_utf8_likely(is_ascii(input))) {
      this->error |= this->prev_incomplete;
    } else {
      // you might think that a for-loop would work, but under Visual Studio, it
      // is not good enough.
      static_assert((simd8x64<uint8_t>::NUM_CHUNKS == 2) ||
                        (simd8x64<uint8_t>::NUM_CHUNKS == 4),
                    "We support either two or four chunks per 64-byte block.");
      if (simd8x64<uint8_t>::NUM_CHUNKS == 2) {
        this->check_utf8_bytes(input.chunks[0], this->prev_input_block);
        this->check_utf8_bytes(input.chunks[1], input.chunks[0]);
      } else if (simd8x64<uint8_t>::NUM_CHUNKS == 4) {
        this->check_utf8_bytes(input.chunks[0], this->prev_input_block);
        this->check_utf8_bytes(input.chunks[1], input.chunks[0]);
        this->check_utf8_bytes(input.chunks[2], input.chunks[1]);
        this->check_utf8_bytes(input.chunks[3], input.chunks[2]);
      }
      this->prev_incomplete =
          is_incomplete(input.chunks[simd8x64<uint8_t>::NUM_CHUNKS - 1]);
      this->prev_input_block = input.chunks[simd8x64<uint8_t>::NUM_CHUNKS - 1];
    }
  }

  // do not forget to call check_eof!
  is_utf8_really_inline bool errors() const {
    return this->error.any_bits_set_anywhere();
  }

}; // struct utf8_checker
} // namespace utf8_validation

using utf8_validation::utf8_checker;

} // unnamed namespace
} // namespace arm64
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace arm64 {
namespace {
namespace utf8_validation {

/**
 * Validates that the string is actual UTF-8.
 */
template <class checker>
bool generic_validate_utf8(const uint8_t *input, size_t length) {
  checker c{};
  buf_block_reader<64> reader(input, length);
  while (reader.has_full_block()) {
    simd::simd8x64<uint8_t> in(reader.full_block());
    c.check_next_input(in);
    reader.advance();
  }
  uint8_t block[64]{};
  reader.get_remainder(block);
  simd::simd8x64<uint8_t> in(block);
  c.check_next_input(in);
  reader.advance();
  c.check_eof();
  return !c.errors();
}

bool generic_validate_utf8(const char *input, size_t length) {
  return generic_validate_utf8<utf8_checker>(
      reinterpret_cast<const uint8_t *>(input), length);
}

} // namespace utf8_validation
} // unnamed namespace
} // namespace arm64
} // namespace is_utf8_internals

//
// Implementation-specific overrides
//
namespace is_utf8_internals {
namespace arm64 {

is_utf8_warn_unused bool
implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return arm64::utf8_validation::generic_validate_utf8(buf, len);
}

} // namespace arm64
} // namespace is_utf8_internals

#endif
#if IS_UTF8_IMPLEMENTATION_FALLBACK

// redefining IS_UTF8_IMPLEMENTATION to "fallback"
// #define IS_UTF8_IMPLEMENTATION fallback

namespace is_utf8_internals {
namespace fallback {

is_utf8_warn_unused bool
implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return scalar::utf8::validate(buf, len);
}

} // namespace fallback
} // namespace is_utf8_internals

#endif
#if IS_UTF8_IMPLEMENTATION_ICELAKE

// redefining IS_UTF8_IMPLEMENTATION to "icelake"
// #define IS_UTF8_IMPLEMENTATION icelake
IS_UTF8_TARGET_ICELAKE

#if IS_UTF8_GCC11ORMORE // workaround for
                        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
IS_UTF8_DISABLE_GCC_WARNING(-Wmaybe-uninitialized)
#endif // end of workaround

namespace is_utf8_internals {
namespace icelake {
namespace {
#ifndef IS_UTF8_ICELAKE_H
#error "icelake.h must be included"
#endif

/**
 * Store the last N bytes of previous followed by 512-N bytes from input.
 */
template <int N> __m512i prev(__m512i input, __m512i previous) {
  static_assert(N <= 32, "N must be no larger than 32");
  const __m512i movemask =
      _mm512_setr_epi32(28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
  const __m512i rotated = _mm512_permutex2var_epi32(input, movemask, previous);
#if IS_UTF8_GCC8 || IS_UTF8_GCC9
  constexpr int shift = 16 - N; // workaround for GCC8,9
  return _mm512_alignr_epi8(input, rotated, shift);
#else
  return _mm512_alignr_epi8(input, rotated, 16 - N);
#endif // IS_UTF8_GCC8 || IS_UTF8_GCC9
}

is_utf8_really_inline __m512i check_special_cases(__m512i input,
                                                  const __m512i prev1) {
  __m512i mask1 = _mm512_setr_epi64(0x0202020202020202, 0x4915012180808080,
                                    0x0202020202020202, 0x4915012180808080,
                                    0x0202020202020202, 0x4915012180808080,
                                    0x0202020202020202, 0x4915012180808080);
  const __m512i v_0f = _mm512_set1_epi8(0x0f);
  __m512i index1 = _mm512_and_si512(_mm512_srli_epi16(prev1, 4), v_0f);

  __m512i byte_1_high = _mm512_shuffle_epi8(mask1, index1);
  __m512i mask2 = _mm512_setr_epi64(0xcbcbcb8b8383a3e7, 0xcbcbdbcbcbcbcbcb,
                                    0xcbcbcb8b8383a3e7, 0xcbcbdbcbcbcbcbcb,
                                    0xcbcbcb8b8383a3e7, 0xcbcbdbcbcbcbcbcb,
                                    0xcbcbcb8b8383a3e7, 0xcbcbdbcbcbcbcbcb);
  __m512i index2 = _mm512_and_si512(prev1, v_0f);

  __m512i byte_1_low = _mm512_shuffle_epi8(mask2, index2);
  __m512i mask3 =
      _mm512_setr_epi64(0x101010101010101, 0x1010101babaaee6, 0x101010101010101,
                        0x1010101babaaee6, 0x101010101010101, 0x1010101babaaee6,
                        0x101010101010101, 0x1010101babaaee6);
  __m512i index3 = _mm512_and_si512(_mm512_srli_epi16(input, 4), v_0f);
  __m512i byte_2_high = _mm512_shuffle_epi8(mask3, index3);
  return _mm512_ternarylogic_epi64(byte_1_high, byte_1_low, byte_2_high, 128);
}

is_utf8_really_inline __m512i check_multibyte_lengths(const __m512i input,
                                                      const __m512i prev_input,
                                                      const __m512i sc) {
  __m512i prev2 = prev<2>(input, prev_input);
  __m512i prev3 = prev<3>(input, prev_input);
  __m512i is_third_byte = _mm512_subs_epu8(
      prev2, _mm512_set1_epi8(char(0b11011111))); // Only 111_____ will be > 0
  __m512i is_fourth_byte = _mm512_subs_epu8(
      prev3, _mm512_set1_epi8(char(0b11101111))); // Only 1111____ will be > 0
  __m512i is_third_or_fourth_byte =
      _mm512_or_si512(is_third_byte, is_fourth_byte);
  const __m512i v_7f = _mm512_set1_epi8(char(0x7f));
  is_third_or_fourth_byte = _mm512_adds_epu8(v_7f, is_third_or_fourth_byte);
  // We want to compute (is_third_or_fourth_byte AND v80) XOR sc.
  const __m512i v_80 = _mm512_set1_epi8(char(0x80));
  return _mm512_ternarylogic_epi32(is_third_or_fourth_byte, v_80, sc,
                                   0b1101010);
  //__m512i is_third_or_fourth_byte_mask =
  //_mm512_and_si512(is_third_or_fourth_byte, v_80); return
  // _mm512_xor_si512(is_third_or_fourth_byte_mask, sc);
}
//
// Return nonzero if there are incomplete multibyte characters at the end of the
// block: e.g. if there is a 4-byte character, but it's 3 bytes from the end.
//
is_utf8_really_inline __m512i is_incomplete(const __m512i input) {
  // If the previous input's last 3 bytes match this, they're too short (they
  // ended at EOF):
  // ... 1111____ 111_____ 11______
  __m512i max_value = _mm512_setr_epi64(0xffffffffffffffff, 0xffffffffffffffff,
                                        0xffffffffffffffff, 0xffffffffffffffff,
                                        0xffffffffffffffff, 0xffffffffffffffff,
                                        0xffffffffffffffff, 0xbfdfefffffffffff);
  return _mm512_subs_epu8(input, max_value);
}

struct avx512_utf8_checker {
  // If this is nonzero, there has been a UTF-8 error.
  __m512i error{};

  // The last input we received
  __m512i prev_input_block{};
  // Whether the last input we received was incomplete (used for ASCII fast
  // path)
  __m512i prev_incomplete{};

  //
  // Check whether the current bytes are valid UTF-8.
  //
  is_utf8_really_inline void check_utf8_bytes(const __m512i input,
                                              const __m512i prev_input) {
    // Flip prev1...prev3 so we can easily determine if they are 2+, 3+ or 4+
    // lead bytes (2, 3, 4-byte leads become large positive numbers instead of
    // small negative numbers)
    __m512i prev1 = prev<1>(input, prev_input);
    __m512i sc = check_special_cases(input, prev1);
    this->error = _mm512_or_si512(
        check_multibyte_lengths(input, prev_input, sc), this->error);
  }

  // The only problem that can happen at EOF is that a multibyte character is
  // too short or a byte value too large in the last bytes: check_special_cases
  // only checks for bytes too large in the first of two bytes.
  is_utf8_really_inline void check_eof() {
    // If the previous block had incomplete UTF-8 characters at the end, an
    // ASCII block can't possibly finish them.
    this->error = _mm512_or_si512(this->error, this->prev_incomplete);
  }

  // returns true if ASCII.
  is_utf8_really_inline bool check_next_input(const __m512i input) {
    const __m512i v_80 = _mm512_set1_epi8(char(0x80));
    const __mmask64 ascii = _mm512_test_epi8_mask(input, v_80);
    if (ascii == 0) {
      this->error = _mm512_or_si512(this->error, this->prev_incomplete);
      return true;
    } else {
      this->check_utf8_bytes(input, this->prev_input_block);
      this->prev_incomplete = is_incomplete(input);
      this->prev_input_block = input;
      return false;
    }
  }
  // do not forget to call check_eof!
  is_utf8_really_inline bool errors() const {
    return _mm512_test_epi8_mask(this->error, this->error) != 0;
  }

}; // struct avx512_utf8_checker

} // namespace
} // namespace icelake
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace icelake {

is_utf8_warn_unused bool
implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  avx512_utf8_checker checker{};
  const char *ptr = buf;
  const char *end = ptr + len;
  for (; ptr + 64 <= end; ptr += 64) {
    const __m512i utf8 = _mm512_loadu_si512((const __m512i *)ptr);
    checker.check_next_input(utf8);
  }
  {
    const __m512i utf8 = _mm512_maskz_loadu_epi8((1ULL << (end - ptr)) - 1,
                                                 (const __m512i *)ptr);
    checker.check_next_input(utf8);
  }
  checker.check_eof();
  return !checker.errors();
}

} // namespace icelake
} // namespace is_utf8_internals

IS_UTF8_UNTARGET_REGION

#if IS_UTF8_GCC11ORMORE // workaround for
                        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
IS_UTF8_POP_DISABLE_WARNINGS
#endif // end of workaround

#endif
#if IS_UTF8_IMPLEMENTATION_HASWELL

// redefining IS_UTF8_IMPLEMENTATION to "haswell"
// #define IS_UTF8_IMPLEMENTATION haswell
IS_UTF8_TARGET_HASWELL

#if IS_UTF8_GCC11ORMORE // workaround for
                        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
IS_UTF8_DISABLE_GCC_WARNING(-Wmaybe-uninitialized)
#endif // end of workaround

namespace is_utf8_internals {
namespace haswell {
namespace {
#ifndef IS_UTF8_HASWELL_H
#error "haswell.h must be included"
#endif
using namespace simd;

is_utf8_really_inline bool is_ascii(const simd8x64<uint8_t> &input) {
  return input.reduce_or().is_ascii();
}

is_utf8_unused is_utf8_really_inline simd8<bool>
must_be_continuation(const simd8<uint8_t> prev1, const simd8<uint8_t> prev2,
                     const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_second_byte =
      prev1.saturating_sub(0b11000000u - 1); // Only 11______ will be > 0
  simd8<uint8_t> is_third_byte =
      prev2.saturating_sub(0b11100000u - 1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte =
      prev3.saturating_sub(0b11110000u - 1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction
  // will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_second_byte | is_third_byte | is_fourth_byte) >
         int8_t(0);
}

is_utf8_really_inline simd8<bool>
must_be_2_3_continuation(const simd8<uint8_t> prev2,
                         const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_third_byte =
      prev2.saturating_sub(0b11100000u - 1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte =
      prev3.saturating_sub(0b11110000u - 1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction
  // will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_third_byte | is_fourth_byte) > int8_t(0);
}

} // unnamed namespace
} // namespace haswell
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace haswell {
namespace {

// Walks through a buffer in block-sized increments, loading the last part with
// spaces
template <size_t STEP_SIZE> struct buf_block_reader {
public:
  is_utf8_really_inline buf_block_reader(const uint8_t *_buf, size_t _len);
  is_utf8_really_inline size_t block_index();
  is_utf8_really_inline bool has_full_block() const;
  is_utf8_really_inline const uint8_t *full_block() const;
  /**
   * Get the last block, padded with spaces.
   *
   * There will always be a last block, with at least 1 byte, unless len == 0
   * (in which case this function fills the buffer with spaces and returns 0. In
   * particular, if len == STEP_SIZE there will be 0 full_blocks and 1 remainder
   * block with STEP_SIZE bytes and no spaces for padding.
   *
   * @return the number of effective characters in the last block.
   */
  is_utf8_really_inline size_t get_remainder(uint8_t *dst) const;
  is_utf8_really_inline void advance();

private:
  const uint8_t *buf;
  const size_t len;
  const size_t lenminusstep;
  size_t idx;
};

// Routines to print masks and text for debugging bitmask operations
is_utf8_unused static char *format_input_text_64(const uint8_t *text) {
  static char *buf =
      reinterpret_cast<char *>(malloc(sizeof(simd8x64<uint8_t>) + 1));
  for (size_t i = 0; i < sizeof(simd8x64<uint8_t>); i++) {
    buf[i] = int8_t(text[i]) < ' ' ? '_' : int8_t(text[i]);
  }
  buf[sizeof(simd8x64<uint8_t>)] = '\0';
  return buf;
}

// Routines to print masks and text for debugging bitmask operations
is_utf8_unused static char *format_input_text(const simd8x64<uint8_t> &in) {
  static char *buf =
      reinterpret_cast<char *>(malloc(sizeof(simd8x64<uint8_t>) + 1));
  in.store(reinterpret_cast<uint8_t *>(buf));
  for (size_t i = 0; i < sizeof(simd8x64<uint8_t>); i++) {
    if (buf[i] < ' ') {
      buf[i] = '_';
    }
  }
  buf[sizeof(simd8x64<uint8_t>)] = '\0';
  return buf;
}

is_utf8_unused static char *format_mask(uint64_t mask) {
  static char *buf = reinterpret_cast<char *>(malloc(64 + 1));
  for (size_t i = 0; i < 64; i++) {
    buf[i] = (mask & (size_t(1) << i)) ? 'X' : ' ';
  }
  buf[64] = '\0';
  return buf;
}

template <size_t STEP_SIZE>
is_utf8_really_inline
buf_block_reader<STEP_SIZE>::buf_block_reader(const uint8_t *_buf, size_t _len)
    : buf{_buf}, len{_len},
      lenminusstep{len < STEP_SIZE ? 0 : len - STEP_SIZE}, idx{0} {}

template <size_t STEP_SIZE>
is_utf8_really_inline size_t buf_block_reader<STEP_SIZE>::block_index() {
  return idx;
}

template <size_t STEP_SIZE>
is_utf8_really_inline bool buf_block_reader<STEP_SIZE>::has_full_block() const {
  return idx < lenminusstep;
}

template <size_t STEP_SIZE>
is_utf8_really_inline const uint8_t *
buf_block_reader<STEP_SIZE>::full_block() const {
  return &buf[idx];
}

template <size_t STEP_SIZE>
is_utf8_really_inline size_t
buf_block_reader<STEP_SIZE>::get_remainder(uint8_t *dst) const {
  if (len == idx) {
    return 0;
  } // memcpy(dst, null, 0) will trigger an error with some sanitizers
  std::memset(dst, 0x20,
              STEP_SIZE); // std::memset STEP_SIZE because it's more efficient
                          // to write out 8 or 16 bytes at once.
  std::memcpy(dst, buf + idx, len - idx);
  return len - idx;
}

template <size_t STEP_SIZE>
is_utf8_really_inline void buf_block_reader<STEP_SIZE>::advance() {
  idx += STEP_SIZE;
}

} // unnamed namespace
} // namespace haswell
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace haswell {
namespace {
namespace utf8_validation {

using namespace simd;

is_utf8_really_inline simd8<uint8_t>
check_special_cases(const simd8<uint8_t> input, const simd8<uint8_t> prev1) {
  // Bit 0 = Too Short (lead byte/ASCII followed by lead byte/ASCII)
  // Bit 1 = Too Long (ASCII followed by continuation)
  // Bit 2 = Overlong 3-byte
  // Bit 4 = Surrogate
  // Bit 5 = Overlong 2-byte
  // Bit 7 = Two Continuations
  constexpr const uint8_t TOO_SHORT = 1 << 0;  // 11______ 0_______
                                               // 11______ 11______
  constexpr const uint8_t TOO_LONG = 1 << 1;   // 0_______ 10______
  constexpr const uint8_t OVERLONG_3 = 1 << 2; // 11100000 100_____
  constexpr const uint8_t SURROGATE = 1 << 4;  // 11101101 101_____
  constexpr const uint8_t OVERLONG_2 = 1 << 5; // 1100000_ 10______
  constexpr const uint8_t TWO_CONTS = 1 << 7;  // 10______ 10______
  constexpr const uint8_t TOO_LARGE = 1 << 3;  // 11110100 1001____
                                               // 11110100 101_____
                                               // 11110101 1001____
                                               // 11110101 101_____
                                               // 1111011_ 1001____
                                               // 1111011_ 101_____
                                               // 11111___ 1001____
                                               // 11111___ 101_____
  constexpr const uint8_t TOO_LARGE_1000 = 1 << 6;
  // 11110101 1000____
  // 1111011_ 1000____
  // 11111___ 1000____
  constexpr const uint8_t OVERLONG_4 = 1 << 6; // 11110000 1000____

  const simd8<uint8_t> byte_1_high = prev1.shr<4>().lookup_16<uint8_t>(
      // 0_______ ________ <ASCII in byte 1>
      TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
      TOO_LONG,
      // 10______ ________ <continuation in byte 1>
      TWO_CONTS, TWO_CONTS, TWO_CONTS, TWO_CONTS,
      // 1100____ ________ <two byte lead in byte 1>
      TOO_SHORT | OVERLONG_2,
      // 1101____ ________ <two byte lead in byte 1>
      TOO_SHORT,
      // 1110____ ________ <three byte lead in byte 1>
      TOO_SHORT | OVERLONG_3 | SURROGATE,
      // 1111____ ________ <four+ byte lead in byte 1>
      TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4);
  constexpr const uint8_t CARRY =
      TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 .
  const simd8<uint8_t> byte_1_low =
      (prev1 & 0x0F)
          .lookup_16<uint8_t>(
              // ____0000 ________
              CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
              // ____0001 ________
              CARRY | OVERLONG_2,
              // ____001_ ________
              CARRY, CARRY,

              // ____0100 ________
              CARRY | TOO_LARGE,
              // ____0101 ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              // ____011_ ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,

              // ____1___ ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              // ____1101 ________
              CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000);
  const simd8<uint8_t> byte_2_high = input.shr<4>().lookup_16<uint8_t>(
      // ________ 0_______ <ASCII in byte 2>
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
      TOO_SHORT, TOO_SHORT,

      // ________ 1000____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 |
          OVERLONG_4,
      // ________ 1001____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
      // ________ 101_____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,

      // ________ 11______
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT);
  return (byte_1_high & byte_1_low & byte_2_high);
}
is_utf8_really_inline simd8<uint8_t>
check_multibyte_lengths(const simd8<uint8_t> input,
                        const simd8<uint8_t> prev_input,
                        const simd8<uint8_t> sc) {
  simd8<uint8_t> prev2 = input.prev<2>(prev_input);
  simd8<uint8_t> prev3 = input.prev<3>(prev_input);
  simd8<uint8_t> must23 =
      simd8<uint8_t>(must_be_2_3_continuation(prev2, prev3));
  simd8<uint8_t> must23_80 = must23 & uint8_t(0x80);
  return must23_80 ^ sc;
}

//
// Return nonzero if there are incomplete multibyte characters at the end of the
// block: e.g. if there is a 4-byte character, but it's 3 bytes from the end.
//
is_utf8_really_inline simd8<uint8_t> is_incomplete(const simd8<uint8_t> input) {
  // If the previous input's last 3 bytes match this, they're too short (they
  // ended at EOF):
  // ... 1111____ 111_____ 11______
  static const uint8_t max_array[32] = {255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        0b11110000u - 1,
                                        0b11100000u - 1,
                                        0b11000000u - 1};
  const simd8<uint8_t> max_value(
      &max_array[sizeof(max_array) - sizeof(simd8<uint8_t>)]);
  return input.gt_bits(max_value);
}

struct utf8_checker {
  // If this is nonzero, there has been a UTF-8 error.
  simd8<uint8_t> error;
  // The last input we received
  simd8<uint8_t> prev_input_block;
  // Whether the last input we received was incomplete (used for ASCII fast
  // path)
  simd8<uint8_t> prev_incomplete;

  //
  // Check whether the current bytes are valid UTF-8.
  //
  is_utf8_really_inline void check_utf8_bytes(const simd8<uint8_t> input,
                                              const simd8<uint8_t> prev_input) {
    // Flip prev1...prev3 so we can easily determine if they are 2+, 3+ or 4+
    // lead bytes (2, 3, 4-byte leads become large positive numbers instead of
    // small negative numbers)
    simd8<uint8_t> prev1 = input.prev<1>(prev_input);
    simd8<uint8_t> sc = check_special_cases(input, prev1);
    this->error |= check_multibyte_lengths(input, prev_input, sc);
  }

  // The only problem that can happen at EOF is that a multibyte character is
  // too short or a byte value too large in the last bytes: check_special_cases
  // only checks for bytes too large in the first of two bytes.
  is_utf8_really_inline void check_eof() {
    // If the previous block had incomplete UTF-8 characters at the end, an
    // ASCII block can't possibly finish them.
    this->error |= this->prev_incomplete;
  }

  is_utf8_really_inline void check_next_input(const simd8x64<uint8_t> &input) {
    if (is_utf8_likely(is_ascii(input))) {
      this->error |= this->prev_incomplete;
    } else {
      // you might think that a for-loop would work, but under Visual Studio, it
      // is not good enough.
      static_assert((simd8x64<uint8_t>::NUM_CHUNKS == 2) ||
                        (simd8x64<uint8_t>::NUM_CHUNKS == 4),
                    "We support either two or four chunks per 64-byte block.");
      if (simd8x64<uint8_t>::NUM_CHUNKS == 2) {
        this->check_utf8_bytes(input.chunks[0], this->prev_input_block);
        this->check_utf8_bytes(input.chunks[1], input.chunks[0]);
      } else if (simd8x64<uint8_t>::NUM_CHUNKS == 4) {
        this->check_utf8_bytes(input.chunks[0], this->prev_input_block);
        this->check_utf8_bytes(input.chunks[1], input.chunks[0]);
        this->check_utf8_bytes(input.chunks[2], input.chunks[1]);
        this->check_utf8_bytes(input.chunks[3], input.chunks[2]);
      }
      this->prev_incomplete =
          is_incomplete(input.chunks[simd8x64<uint8_t>::NUM_CHUNKS - 1]);
      this->prev_input_block = input.chunks[simd8x64<uint8_t>::NUM_CHUNKS - 1];
    }
  }

  // do not forget to call check_eof!
  is_utf8_really_inline bool errors() const {
    return this->error.any_bits_set_anywhere();
  }

}; // struct utf8_checker
} // namespace utf8_validation

using utf8_validation::utf8_checker;

} // unnamed namespace
} // namespace haswell
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace haswell {
namespace {
namespace utf8_validation {

/**
 * Validates that the string is actual UTF-8.
 */
template <class checker>
bool generic_validate_utf8(const uint8_t *input, size_t length) {
  checker c{};
  buf_block_reader<64> reader(input, length);
  while (reader.has_full_block()) {
    simd::simd8x64<uint8_t> in(reader.full_block());
    c.check_next_input(in);
    reader.advance();
  }
  uint8_t block[64]{};
  reader.get_remainder(block);
  simd::simd8x64<uint8_t> in(block);
  c.check_next_input(in);
  reader.advance();
  c.check_eof();
  return !c.errors();
}

bool generic_validate_utf8(const char *input, size_t length) {
  return generic_validate_utf8<utf8_checker>(
      reinterpret_cast<const uint8_t *>(input), length);
}

} // namespace utf8_validation
} // unnamed namespace
} // namespace haswell
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace haswell {

is_utf8_warn_unused bool
implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return haswell::utf8_validation::generic_validate_utf8(buf, len);
}

} // namespace haswell
} // namespace is_utf8_internals

IS_UTF8_UNTARGET_REGION

#if IS_UTF8_GCC11ORMORE // workaround for
                        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105593
#pragma GCC diagnostic pop
#endif // end of workaround

#endif
#if IS_UTF8_IMPLEMENTATION_PPC64

// redefining IS_UTF8_IMPLEMENTATION to "ppc64"
// #define IS_UTF8_IMPLEMENTATION ppc64

namespace is_utf8_internals {
namespace ppc64 {
namespace {
#ifndef IS_UTF8_PPC64_H
#error "ppc64.h must be included"
#endif
using namespace simd;

is_utf8_really_inline bool is_ascii(const simd8x64<uint8_t> &input) {
  // careful: 0x80 is not ascii.
  return input.reduce_or().saturating_sub(0b01111111u).bits_not_set_anywhere();
}

is_utf8_unused is_utf8_really_inline simd8<bool>
must_be_continuation(const simd8<uint8_t> prev1, const simd8<uint8_t> prev2,
                     const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_second_byte =
      prev1.saturating_sub(0b11000000u - 1); // Only 11______ will be > 0
  simd8<uint8_t> is_third_byte =
      prev2.saturating_sub(0b11100000u - 1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte =
      prev3.saturating_sub(0b11110000u - 1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction
  // will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_second_byte | is_third_byte | is_fourth_byte) >
         int8_t(0);
}

is_utf8_really_inline simd8<bool>
must_be_2_3_continuation(const simd8<uint8_t> prev2,
                         const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_third_byte =
      prev2.saturating_sub(0b11100000u - 1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte =
      prev3.saturating_sub(0b11110000u - 1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction
  // will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_third_byte | is_fourth_byte) > int8_t(0);
}

} // unnamed namespace
} // namespace ppc64
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace ppc64 {
namespace {

// Walks through a buffer in block-sized increments, loading the last part with
// spaces
template <size_t STEP_SIZE> struct buf_block_reader {
public:
  is_utf8_really_inline buf_block_reader(const uint8_t *_buf, size_t _len);
  is_utf8_really_inline size_t block_index();
  is_utf8_really_inline bool has_full_block() const;
  is_utf8_really_inline const uint8_t *full_block() const;
  /**
   * Get the last block, padded with spaces.
   *
   * There will always be a last block, with at least 1 byte, unless len == 0
   * (in which case this function fills the buffer with spaces and returns 0. In
   * particular, if len == STEP_SIZE there will be 0 full_blocks and 1 remainder
   * block with STEP_SIZE bytes and no spaces for padding.
   *
   * @return the number of effective characters in the last block.
   */
  is_utf8_really_inline size_t get_remainder(uint8_t *dst) const;
  is_utf8_really_inline void advance();

private:
  const uint8_t *buf;
  const size_t len;
  const size_t lenminusstep;
  size_t idx;
};

// Routines to print masks and text for debugging bitmask operations
is_utf8_unused static char *format_input_text_64(const uint8_t *text) {
  static char *buf =
      reinterpret_cast<char *>(malloc(sizeof(simd8x64<uint8_t>) + 1));
  for (size_t i = 0; i < sizeof(simd8x64<uint8_t>); i++) {
    buf[i] = int8_t(text[i]) < ' ' ? '_' : int8_t(text[i]);
  }
  buf[sizeof(simd8x64<uint8_t>)] = '\0';
  return buf;
}

// Routines to print masks and text for debugging bitmask operations
is_utf8_unused static char *format_input_text(const simd8x64<uint8_t> &in) {
  static char *buf =
      reinterpret_cast<char *>(malloc(sizeof(simd8x64<uint8_t>) + 1));
  in.store(reinterpret_cast<uint8_t *>(buf));
  for (size_t i = 0; i < sizeof(simd8x64<uint8_t>); i++) {
    if (buf[i] < ' ') {
      buf[i] = '_';
    }
  }
  buf[sizeof(simd8x64<uint8_t>)] = '\0';
  return buf;
}

is_utf8_unused static char *format_mask(uint64_t mask) {
  static char *buf = reinterpret_cast<char *>(malloc(64 + 1));
  for (size_t i = 0; i < 64; i++) {
    buf[i] = (mask & (size_t(1) << i)) ? 'X' : ' ';
  }
  buf[64] = '\0';
  return buf;
}

template <size_t STEP_SIZE>
is_utf8_really_inline
buf_block_reader<STEP_SIZE>::buf_block_reader(const uint8_t *_buf, size_t _len)
    : buf{_buf}, len{_len},
      lenminusstep{len < STEP_SIZE ? 0 : len - STEP_SIZE}, idx{0} {}

template <size_t STEP_SIZE>
is_utf8_really_inline size_t buf_block_reader<STEP_SIZE>::block_index() {
  return idx;
}

template <size_t STEP_SIZE>
is_utf8_really_inline bool buf_block_reader<STEP_SIZE>::has_full_block() const {
  return idx < lenminusstep;
}

template <size_t STEP_SIZE>
is_utf8_really_inline const uint8_t *
buf_block_reader<STEP_SIZE>::full_block() const {
  return &buf[idx];
}

template <size_t STEP_SIZE>
is_utf8_really_inline size_t
buf_block_reader<STEP_SIZE>::get_remainder(uint8_t *dst) const {
  if (len == idx) {
    return 0;
  } // memcpy(dst, null, 0) will trigger an error with some sanitizers
  std::memset(dst, 0x20,
              STEP_SIZE); // std::memset STEP_SIZE because it's more efficient
                          // to write out 8 or 16 bytes at once.
  std::memcpy(dst, buf + idx, len - idx);
  return len - idx;
}

template <size_t STEP_SIZE>
is_utf8_really_inline void buf_block_reader<STEP_SIZE>::advance() {
  idx += STEP_SIZE;
}

} // unnamed namespace
} // namespace ppc64
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace ppc64 {
namespace {
namespace utf8_validation {

using namespace simd;

is_utf8_really_inline simd8<uint8_t>
check_special_cases(const simd8<uint8_t> input, const simd8<uint8_t> prev1) {
  // Bit 0 = Too Short (lead byte/ASCII followed by lead byte/ASCII)
  // Bit 1 = Too Long (ASCII followed by continuation)
  // Bit 2 = Overlong 3-byte
  // Bit 4 = Surrogate
  // Bit 5 = Overlong 2-byte
  // Bit 7 = Two Continuations
  constexpr const uint8_t TOO_SHORT = 1 << 0;  // 11______ 0_______
                                               // 11______ 11______
  constexpr const uint8_t TOO_LONG = 1 << 1;   // 0_______ 10______
  constexpr const uint8_t OVERLONG_3 = 1 << 2; // 11100000 100_____
  constexpr const uint8_t SURROGATE = 1 << 4;  // 11101101 101_____
  constexpr const uint8_t OVERLONG_2 = 1 << 5; // 1100000_ 10______
  constexpr const uint8_t TWO_CONTS = 1 << 7;  // 10______ 10______
  constexpr const uint8_t TOO_LARGE = 1 << 3;  // 11110100 1001____
                                               // 11110100 101_____
                                               // 11110101 1001____
                                               // 11110101 101_____
                                               // 1111011_ 1001____
                                               // 1111011_ 101_____
                                               // 11111___ 1001____
                                               // 11111___ 101_____
  constexpr const uint8_t TOO_LARGE_1000 = 1 << 6;
  // 11110101 1000____
  // 1111011_ 1000____
  // 11111___ 1000____
  constexpr const uint8_t OVERLONG_4 = 1 << 6; // 11110000 1000____

  const simd8<uint8_t> byte_1_high = prev1.shr<4>().lookup_16<uint8_t>(
      // 0_______ ________ <ASCII in byte 1>
      TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
      TOO_LONG,
      // 10______ ________ <continuation in byte 1>
      TWO_CONTS, TWO_CONTS, TWO_CONTS, TWO_CONTS,
      // 1100____ ________ <two byte lead in byte 1>
      TOO_SHORT | OVERLONG_2,
      // 1101____ ________ <two byte lead in byte 1>
      TOO_SHORT,
      // 1110____ ________ <three byte lead in byte 1>
      TOO_SHORT | OVERLONG_3 | SURROGATE,
      // 1111____ ________ <four+ byte lead in byte 1>
      TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4);
  constexpr const uint8_t CARRY =
      TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 .
  const simd8<uint8_t> byte_1_low =
      (prev1 & 0x0F)
          .lookup_16<uint8_t>(
              // ____0000 ________
              CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
              // ____0001 ________
              CARRY | OVERLONG_2,
              // ____001_ ________
              CARRY, CARRY,

              // ____0100 ________
              CARRY | TOO_LARGE,
              // ____0101 ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              // ____011_ ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,

              // ____1___ ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              // ____1101 ________
              CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000);
  const simd8<uint8_t> byte_2_high = input.shr<4>().lookup_16<uint8_t>(
      // ________ 0_______ <ASCII in byte 2>
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
      TOO_SHORT, TOO_SHORT,

      // ________ 1000____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 |
          OVERLONG_4,
      // ________ 1001____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
      // ________ 101_____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,

      // ________ 11______
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT);
  return (byte_1_high & byte_1_low & byte_2_high);
}
is_utf8_really_inline simd8<uint8_t>
check_multibyte_lengths(const simd8<uint8_t> input,
                        const simd8<uint8_t> prev_input,
                        const simd8<uint8_t> sc) {
  simd8<uint8_t> prev2 = input.prev<2>(prev_input);
  simd8<uint8_t> prev3 = input.prev<3>(prev_input);
  simd8<uint8_t> must23 =
      simd8<uint8_t>(must_be_2_3_continuation(prev2, prev3));
  simd8<uint8_t> must23_80 = must23 & uint8_t(0x80);
  return must23_80 ^ sc;
}

//
// Return nonzero if there are incomplete multibyte characters at the end of the
// block: e.g. if there is a 4-byte character, but it's 3 bytes from the end.
//
is_utf8_really_inline simd8<uint8_t> is_incomplete(const simd8<uint8_t> input) {
  // If the previous input's last 3 bytes match this, they're too short (they
  // ended at EOF):
  // ... 1111____ 111_____ 11______
  static const uint8_t max_array[32] = {255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        0b11110000u - 1,
                                        0b11100000u - 1,
                                        0b11000000u - 1};
  const simd8<uint8_t> max_value(
      &max_array[sizeof(max_array) - sizeof(simd8<uint8_t>)]);
  return input.gt_bits(max_value);
}

struct utf8_checker {
  // If this is nonzero, there has been a UTF-8 error.
  simd8<uint8_t> error;
  // The last input we received
  simd8<uint8_t> prev_input_block;
  // Whether the last input we received was incomplete (used for ASCII fast
  // path)
  simd8<uint8_t> prev_incomplete;

  //
  // Check whether the current bytes are valid UTF-8.
  //
  is_utf8_really_inline void check_utf8_bytes(const simd8<uint8_t> input,
                                              const simd8<uint8_t> prev_input) {
    // Flip prev1...prev3 so we can easily determine if they are 2+, 3+ or 4+
    // lead bytes (2, 3, 4-byte leads become large positive numbers instead of
    // small negative numbers)
    simd8<uint8_t> prev1 = input.prev<1>(prev_input);
    simd8<uint8_t> sc = check_special_cases(input, prev1);
    this->error |= check_multibyte_lengths(input, prev_input, sc);
  }

  // The only problem that can happen at EOF is that a multibyte character is
  // too short or a byte value too large in the last bytes: check_special_cases
  // only checks for bytes too large in the first of two bytes.
  is_utf8_really_inline void check_eof() {
    // If the previous block had incomplete UTF-8 characters at the end, an
    // ASCII block can't possibly finish them.
    this->error |= this->prev_incomplete;
  }

  is_utf8_really_inline void check_next_input(const simd8x64<uint8_t> &input) {
    if (is_utf8_likely(is_ascii(input))) {
      this->error |= this->prev_incomplete;
    } else {
      // you might think that a for-loop would work, but under Visual Studio, it
      // is not good enough.
      static_assert((simd8x64<uint8_t>::NUM_CHUNKS == 2) ||
                        (simd8x64<uint8_t>::NUM_CHUNKS == 4),
                    "We support either two or four chunks per 64-byte block.");
      if (simd8x64<uint8_t>::NUM_CHUNKS == 2) {
        this->check_utf8_bytes(input.chunks[0], this->prev_input_block);
        this->check_utf8_bytes(input.chunks[1], input.chunks[0]);
      } else if (simd8x64<uint8_t>::NUM_CHUNKS == 4) {
        this->check_utf8_bytes(input.chunks[0], this->prev_input_block);
        this->check_utf8_bytes(input.chunks[1], input.chunks[0]);
        this->check_utf8_bytes(input.chunks[2], input.chunks[1]);
        this->check_utf8_bytes(input.chunks[3], input.chunks[2]);
      }
      this->prev_incomplete =
          is_incomplete(input.chunks[simd8x64<uint8_t>::NUM_CHUNKS - 1]);
      this->prev_input_block = input.chunks[simd8x64<uint8_t>::NUM_CHUNKS - 1];
    }
  }

  // do not forget to call check_eof!
  is_utf8_really_inline bool errors() const {
    return this->error.any_bits_set_anywhere();
  }

}; // struct utf8_checker
} // namespace utf8_validation

using utf8_validation::utf8_checker;

} // unnamed namespace
} // namespace ppc64
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace ppc64 {
namespace {
namespace utf8_validation {

/**
 * Validates that the string is actual UTF-8.
 */
template <class checker>
bool generic_validate_utf8(const uint8_t *input, size_t length) {
  checker c{};
  buf_block_reader<64> reader(input, length);
  while (reader.has_full_block()) {
    simd::simd8x64<uint8_t> in(reader.full_block());
    c.check_next_input(in);
    reader.advance();
  }
  uint8_t block[64]{};
  reader.get_remainder(block);
  simd::simd8x64<uint8_t> in(block);
  c.check_next_input(in);
  reader.advance();
  c.check_eof();
  return !c.errors();
}

bool generic_validate_utf8(const char *input, size_t length) {
  return generic_validate_utf8<utf8_checker>(
      reinterpret_cast<const uint8_t *>(input), length);
}

} // namespace utf8_validation
} // unnamed namespace
} // namespace ppc64
} // namespace is_utf8_internals

//
// Implementation-specific overrides
//
namespace is_utf8_internals {
namespace ppc64 {

is_utf8_warn_unused bool
implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return ppc64::utf8_validation::generic_validate_utf8(buf, len);
}

} // namespace ppc64
} // namespace is_utf8_internals

#endif
#if IS_UTF8_IMPLEMENTATION_WESTMERE

// redefining IS_UTF8_IMPLEMENTATION to "westmere"
// #define IS_UTF8_IMPLEMENTATION westmere
IS_UTF8_TARGET_WESTMERE

namespace is_utf8_internals {
namespace westmere {
namespace {
#ifndef IS_UTF8_WESTMERE_H
#error "westmere.h must be included"
#endif
using namespace simd;

is_utf8_really_inline bool is_ascii(const simd8x64<uint8_t> &input) {
  return input.reduce_or().is_ascii();
}

is_utf8_unused is_utf8_really_inline simd8<bool>
must_be_continuation(const simd8<uint8_t> prev1, const simd8<uint8_t> prev2,
                     const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_second_byte =
      prev1.saturating_sub(0b11000000u - 1); // Only 11______ will be > 0
  simd8<uint8_t> is_third_byte =
      prev2.saturating_sub(0b11100000u - 1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte =
      prev3.saturating_sub(0b11110000u - 1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction
  // will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_second_byte | is_third_byte | is_fourth_byte) >
         int8_t(0);
}

is_utf8_really_inline simd8<bool>
must_be_2_3_continuation(const simd8<uint8_t> prev2,
                         const simd8<uint8_t> prev3) {
  simd8<uint8_t> is_third_byte =
      prev2.saturating_sub(0b11100000u - 1); // Only 111_____ will be > 0
  simd8<uint8_t> is_fourth_byte =
      prev3.saturating_sub(0b11110000u - 1); // Only 1111____ will be > 0
  // Caller requires a bool (all 1's). All values resulting from the subtraction
  // will be <= 64, so signed comparison is fine.
  return simd8<int8_t>(is_third_byte | is_fourth_byte) > int8_t(0);
}

} // unnamed namespace
} // namespace westmere
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace westmere {
namespace {

// Walks through a buffer in block-sized increments, loading the last part with
// spaces
template <size_t STEP_SIZE> struct buf_block_reader {
public:
  is_utf8_really_inline buf_block_reader(const uint8_t *_buf, size_t _len);
  is_utf8_really_inline size_t block_index();
  is_utf8_really_inline bool has_full_block() const;
  is_utf8_really_inline const uint8_t *full_block() const;
  /**
   * Get the last block, padded with spaces.
   *
   * There will always be a last block, with at least 1 byte, unless len == 0
   * (in which case this function fills the buffer with spaces and returns 0. In
   * particular, if len == STEP_SIZE there will be 0 full_blocks and 1 remainder
   * block with STEP_SIZE bytes and no spaces for padding.
   *
   * @return the number of effective characters in the last block.
   */
  is_utf8_really_inline size_t get_remainder(uint8_t *dst) const;
  is_utf8_really_inline void advance();

private:
  const uint8_t *buf;
  const size_t len;
  const size_t lenminusstep;
  size_t idx;
};

// Routines to print masks and text for debugging bitmask operations
is_utf8_unused static char *format_input_text_64(const uint8_t *text) {
  static char *buf =
      reinterpret_cast<char *>(malloc(sizeof(simd8x64<uint8_t>) + 1));
  for (size_t i = 0; i < sizeof(simd8x64<uint8_t>); i++) {
    buf[i] = int8_t(text[i]) < ' ' ? '_' : int8_t(text[i]);
  }
  buf[sizeof(simd8x64<uint8_t>)] = '\0';
  return buf;
}

// Routines to print masks and text for debugging bitmask operations
is_utf8_unused static char *format_input_text(const simd8x64<uint8_t> &in) {
  static char *buf =
      reinterpret_cast<char *>(malloc(sizeof(simd8x64<uint8_t>) + 1));
  in.store(reinterpret_cast<uint8_t *>(buf));
  for (size_t i = 0; i < sizeof(simd8x64<uint8_t>); i++) {
    if (buf[i] < ' ') {
      buf[i] = '_';
    }
  }
  buf[sizeof(simd8x64<uint8_t>)] = '\0';
  return buf;
}

is_utf8_unused static char *format_mask(uint64_t mask) {
  static char *buf = reinterpret_cast<char *>(malloc(64 + 1));
  for (size_t i = 0; i < 64; i++) {
    buf[i] = (mask & (size_t(1) << i)) ? 'X' : ' ';
  }
  buf[64] = '\0';
  return buf;
}

template <size_t STEP_SIZE>
is_utf8_really_inline
buf_block_reader<STEP_SIZE>::buf_block_reader(const uint8_t *_buf, size_t _len)
    : buf{_buf}, len{_len},
      lenminusstep{len < STEP_SIZE ? 0 : len - STEP_SIZE}, idx{0} {}

template <size_t STEP_SIZE>
is_utf8_really_inline size_t buf_block_reader<STEP_SIZE>::block_index() {
  return idx;
}

template <size_t STEP_SIZE>
is_utf8_really_inline bool buf_block_reader<STEP_SIZE>::has_full_block() const {
  return idx < lenminusstep;
}

template <size_t STEP_SIZE>
is_utf8_really_inline const uint8_t *
buf_block_reader<STEP_SIZE>::full_block() const {
  return &buf[idx];
}

template <size_t STEP_SIZE>
is_utf8_really_inline size_t
buf_block_reader<STEP_SIZE>::get_remainder(uint8_t *dst) const {
  if (len == idx) {
    return 0;
  } // memcpy(dst, null, 0) will trigger an error with some sanitizers
  std::memset(dst, 0x20,
              STEP_SIZE); // std::memset STEP_SIZE because it's more efficient
                          // to write out 8 or 16 bytes at once.
  std::memcpy(dst, buf + idx, len - idx);
  return len - idx;
}

template <size_t STEP_SIZE>
is_utf8_really_inline void buf_block_reader<STEP_SIZE>::advance() {
  idx += STEP_SIZE;
}

} // unnamed namespace
} // namespace westmere
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace westmere {
namespace {
namespace utf8_validation {

using namespace simd;

is_utf8_really_inline simd8<uint8_t>
check_special_cases(const simd8<uint8_t> input, const simd8<uint8_t> prev1) {
  // Bit 0 = Too Short (lead byte/ASCII followed by lead byte/ASCII)
  // Bit 1 = Too Long (ASCII followed by continuation)
  // Bit 2 = Overlong 3-byte
  // Bit 4 = Surrogate
  // Bit 5 = Overlong 2-byte
  // Bit 7 = Two Continuations
  constexpr const uint8_t TOO_SHORT = 1 << 0;  // 11______ 0_______
                                               // 11______ 11______
  constexpr const uint8_t TOO_LONG = 1 << 1;   // 0_______ 10______
  constexpr const uint8_t OVERLONG_3 = 1 << 2; // 11100000 100_____
  constexpr const uint8_t SURROGATE = 1 << 4;  // 11101101 101_____
  constexpr const uint8_t OVERLONG_2 = 1 << 5; // 1100000_ 10______
  constexpr const uint8_t TWO_CONTS = 1 << 7;  // 10______ 10______
  constexpr const uint8_t TOO_LARGE = 1 << 3;  // 11110100 1001____
                                               // 11110100 101_____
                                               // 11110101 1001____
                                               // 11110101 101_____
                                               // 1111011_ 1001____
                                               // 1111011_ 101_____
                                               // 11111___ 1001____
                                               // 11111___ 101_____
  constexpr const uint8_t TOO_LARGE_1000 = 1 << 6;
  // 11110101 1000____
  // 1111011_ 1000____
  // 11111___ 1000____
  constexpr const uint8_t OVERLONG_4 = 1 << 6; // 11110000 1000____

  const simd8<uint8_t> byte_1_high = prev1.shr<4>().lookup_16<uint8_t>(
      // 0_______ ________ <ASCII in byte 1>
      TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
      TOO_LONG,
      // 10______ ________ <continuation in byte 1>
      TWO_CONTS, TWO_CONTS, TWO_CONTS, TWO_CONTS,
      // 1100____ ________ <two byte lead in byte 1>
      TOO_SHORT | OVERLONG_2,
      // 1101____ ________ <two byte lead in byte 1>
      TOO_SHORT,
      // 1110____ ________ <three byte lead in byte 1>
      TOO_SHORT | OVERLONG_3 | SURROGATE,
      // 1111____ ________ <four+ byte lead in byte 1>
      TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4);
  constexpr const uint8_t CARRY =
      TOO_SHORT | TOO_LONG | TWO_CONTS; // These all have ____ in byte 1 .
  const simd8<uint8_t> byte_1_low =
      (prev1 & 0x0F)
          .lookup_16<uint8_t>(
              // ____0000 ________
              CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
              // ____0001 ________
              CARRY | OVERLONG_2,
              // ____001_ ________
              CARRY, CARRY,

              // ____0100 ________
              CARRY | TOO_LARGE,
              // ____0101 ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              // ____011_ ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,

              // ____1___ ________
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              // ____1101 ________
              CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
              CARRY | TOO_LARGE | TOO_LARGE_1000,
              CARRY | TOO_LARGE | TOO_LARGE_1000);
  const simd8<uint8_t> byte_2_high = input.shr<4>().lookup_16<uint8_t>(
      // ________ 0_______ <ASCII in byte 2>
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
      TOO_SHORT, TOO_SHORT,

      // ________ 1000____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 |
          OVERLONG_4,
      // ________ 1001____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
      // ________ 101_____
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
      TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,

      // ________ 11______
      TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT);
  return (byte_1_high & byte_1_low & byte_2_high);
}
is_utf8_really_inline simd8<uint8_t>
check_multibyte_lengths(const simd8<uint8_t> input,
                        const simd8<uint8_t> prev_input,
                        const simd8<uint8_t> sc) {
  simd8<uint8_t> prev2 = input.prev<2>(prev_input);
  simd8<uint8_t> prev3 = input.prev<3>(prev_input);
  simd8<uint8_t> must23 =
      simd8<uint8_t>(must_be_2_3_continuation(prev2, prev3));
  simd8<uint8_t> must23_80 = must23 & uint8_t(0x80);
  return must23_80 ^ sc;
}

//
// Return nonzero if there are incomplete multibyte characters at the end of the
// block: e.g. if there is a 4-byte character, but it's 3 bytes from the end.
//
is_utf8_really_inline simd8<uint8_t> is_incomplete(const simd8<uint8_t> input) {
  // If the previous input's last 3 bytes match this, they're too short (they
  // ended at EOF):
  // ... 1111____ 111_____ 11______
  static const uint8_t max_array[32] = {255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        255,
                                        0b11110000u - 1,
                                        0b11100000u - 1,
                                        0b11000000u - 1};
  const simd8<uint8_t> max_value(
      &max_array[sizeof(max_array) - sizeof(simd8<uint8_t>)]);
  return input.gt_bits(max_value);
}

struct utf8_checker {
  // If this is nonzero, there has been a UTF-8 error.
  simd8<uint8_t> error;
  // The last input we received
  simd8<uint8_t> prev_input_block;
  // Whether the last input we received was incomplete (used for ASCII fast
  // path)
  simd8<uint8_t> prev_incomplete;

  //
  // Check whether the current bytes are valid UTF-8.
  //
  is_utf8_really_inline void check_utf8_bytes(const simd8<uint8_t> input,
                                              const simd8<uint8_t> prev_input) {
    // Flip prev1...prev3 so we can easily determine if they are 2+, 3+ or 4+
    // lead bytes (2, 3, 4-byte leads become large positive numbers instead of
    // small negative numbers)
    simd8<uint8_t> prev1 = input.prev<1>(prev_input);
    simd8<uint8_t> sc = check_special_cases(input, prev1);
    this->error |= check_multibyte_lengths(input, prev_input, sc);
  }

  // The only problem that can happen at EOF is that a multibyte character is
  // too short or a byte value too large in the last bytes: check_special_cases
  // only checks for bytes too large in the first of two bytes.
  is_utf8_really_inline void check_eof() {
    // If the previous block had incomplete UTF-8 characters at the end, an
    // ASCII block can't possibly finish them.
    this->error |= this->prev_incomplete;
  }

  is_utf8_really_inline void check_next_input(const simd8x64<uint8_t> &input) {
    if (is_utf8_likely(is_ascii(input))) {
      this->error |= this->prev_incomplete;
    } else {
      // you might think that a for-loop would work, but under Visual Studio, it
      // is not good enough.
      static_assert((simd8x64<uint8_t>::NUM_CHUNKS == 2) ||
                        (simd8x64<uint8_t>::NUM_CHUNKS == 4),
                    "We support either two or four chunks per 64-byte block.");
      if (simd8x64<uint8_t>::NUM_CHUNKS == 2) {
        this->check_utf8_bytes(input.chunks[0], this->prev_input_block);
        this->check_utf8_bytes(input.chunks[1], input.chunks[0]);
      } else if (simd8x64<uint8_t>::NUM_CHUNKS == 4) {
        this->check_utf8_bytes(input.chunks[0], this->prev_input_block);
        this->check_utf8_bytes(input.chunks[1], input.chunks[0]);
        this->check_utf8_bytes(input.chunks[2], input.chunks[1]);
        this->check_utf8_bytes(input.chunks[3], input.chunks[2]);
      }
      this->prev_incomplete =
          is_incomplete(input.chunks[simd8x64<uint8_t>::NUM_CHUNKS - 1]);
      this->prev_input_block = input.chunks[simd8x64<uint8_t>::NUM_CHUNKS - 1];
    }
  }

  // do not forget to call check_eof!
  is_utf8_really_inline bool errors() const {
    return this->error.any_bits_set_anywhere();
  }

}; // struct utf8_checker
} // namespace utf8_validation

using utf8_validation::utf8_checker;

} // unnamed namespace
} // namespace westmere
} // namespace is_utf8_internals

namespace is_utf8_internals {
namespace westmere {
namespace {
namespace utf8_validation {

/**
 * Validates that the string is actual UTF-8.
 */
template <class checker>
bool generic_validate_utf8(const uint8_t *input, size_t length) {
  checker c{};
  buf_block_reader<64> reader(input, length);
  while (reader.has_full_block()) {
    simd::simd8x64<uint8_t> in(reader.full_block());
    c.check_next_input(in);
    reader.advance();
  }
  uint8_t block[64]{};
  reader.get_remainder(block);
  simd::simd8x64<uint8_t> in(block);
  c.check_next_input(in);
  reader.advance();
  c.check_eof();
  return !c.errors();
}

bool generic_validate_utf8(const char *input, size_t length) {
  return generic_validate_utf8<utf8_checker>(
      reinterpret_cast<const uint8_t *>(input), length);
}

} // namespace utf8_validation
} // unnamed namespace
} // namespace westmere
} // namespace is_utf8_internals
//
// Implementation-specific overrides
//

namespace is_utf8_internals {
namespace westmere {

is_utf8_warn_unused bool
implementation::validate_utf8(const char *buf, size_t len) const noexcept {
  return westmere::utf8_validation::generic_validate_utf8(buf, len);
}

} // namespace westmere
} // namespace is_utf8_internals

IS_UTF8_UNTARGET_REGION

#endif

IS_UTF8_POP_DISABLE_WARNINGS

bool is_utf8(const char *src, size_t len) {
  return is_utf8_internals::validate_utf8(src, len);
}
