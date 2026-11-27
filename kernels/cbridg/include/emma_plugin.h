/* EMMA native plugin interface v1. Gavriel Technologies.
 * A plugin is a shared library that exports emma_plugin_v1(). EMMA loads it from
 * <home>/plugins and offers it as a Lego piece. Plain C only (AB rules); see
 * emma-labs/docs/guides/codstd/2026-11-01-platfm-guide-where-c.md.
 * Plugins are trusted code: they run in EMMA's process. */
#ifndef EMMA_PLUGIN_H
#define EMMA_PLUGIN_H
#include <stddef.h>
#include <stdint.h>

#define EMMA_PLUGIN_ABI 1u
/* run statuses */
#define EMMA_PLUGIN_OK 0
#define EMMA_PLUGIN_SMALL 1 /* cap too small; *out_len holds the bytes needed */
#define EMMA_PLUGIN_FAIL -1 /* failed; out holds a UTF-8 message of *out_len bytes */

#if defined(_WIN32)
#define EMMA_PLUGIN_EXPORT __declspec(dllexport)
#else
#define EMMA_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif
/* [in] input: the piece's input values as one UTF-8 JSON object of input_len bytes.
 * [out] out: the output values as one JSON object, at most cap bytes; *out_len: bytes
 * written, or needed with EMMA_PLUGIN_SMALL. Must be safe to call from several threads. */
typedef int32_t (*emma_plugin_run_fn)(void* state, const uint8_t* input, size_t input_len,
                                      uint8_t* out, size_t cap, size_t* out_len);
/* Static table owned by the plugin; valid until the library is unloaded. */
typedef struct emma_plugin_v1 {
  uint32_t abi;      /* EMMA_PLUGIN_ABI */
  uint32_t size;     /* sizeof(emma_plugin_v1) */
  const char* name;  /* piece id, for example "sample.word-count" (UTF-8, NUL-terminated) */
  const char* doc;   /* one-line summary */
  const char* ports; /* {"inputs":[["text","text"]],"outputs":[["words","any"]]} */
  void* state;       /* passed back to run; may be NULL */
  emma_plugin_run_fn run;
} emma_plugin_v1;
/* The one export a plugin library provides. */
typedef const emma_plugin_v1* (*emma_plugin_entry_fn)(void);
EMMA_PLUGIN_EXPORT const emma_plugin_v1* emma_plugin_v1_entry(void);
#ifdef __cplusplus
}
#endif
#endif
