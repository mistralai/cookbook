#define NAPI_VERSION 1
#include <assert.h>
#include <node_api.h>

#include "../deps/is_utf8/include/is_utf8.h"

napi_value IsValidUTF8(napi_env env, napi_callback_info info) {
  napi_status status;
  size_t argc = 1;
  napi_value argv[1];

  status = napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
  assert(status == napi_ok);

  char *buf;
  size_t len;

  status = napi_get_buffer_info(env, argv[0], (void **)&buf, &len);
  assert(status == napi_ok);

  bool is_valid = is_utf8(buf, len);

  napi_value result;
  status = napi_get_boolean(env, is_valid, &result);
  assert(status == napi_ok);

  return result;
}

napi_value Init(napi_env env, napi_value exports) {
  napi_status status;
  napi_value isValidUTF8;

  status = napi_create_function(env, NULL, 0, IsValidUTF8, NULL, &isValidUTF8);
  assert(status == napi_ok);

  return isValidUTF8;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
