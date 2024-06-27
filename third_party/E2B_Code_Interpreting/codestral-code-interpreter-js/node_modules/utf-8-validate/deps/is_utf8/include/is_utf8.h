#ifndef IS_UTF8
#include <stddef.h>

// Check whether the provided string is UTF-8.
// The function is designed for use cases where
// 99.99% of the inputs are valid UTF-8.
// Thus the function unconditionally scans the
// whole input.
bool is_utf8(const char *src, size_t len);
#endif // IS_UTF8