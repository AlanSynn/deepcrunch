#pragma once
//==============================================================================
#ifdef _WIN32
#if defined(deepcrunch_EXPORTS)
#define DEEPCRUNCH_API __declspec(dllexport)
#else
#define DEEPCRUNCH_API __declspec(dllimport)
#endif
#else
#define DEEPCRUNCH_API
#endif
//==============================================================================
#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define DEEPCRUNCH_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define DEEPCRUNCH_INLINE_VARIABLE __declspec(selectany)
#else
#define DEEPCRUNCH_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
//==============================================================================