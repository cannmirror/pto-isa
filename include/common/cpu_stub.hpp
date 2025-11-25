#ifndef PTO_CPUSTUB_HPP
#define PTO_CPUSTUB_HPP

#include <cstdlib>
#include <cstring>
#include <cassert>

#define __global__
#define __aicore__
#define __gm__
#define __out__
#define __in__
#define __ubuf__
#define __cbuf__
#define __ca__
#define __cb__
#define __cc__
#define __fbuf__
#define __tf__

typedef void* aclrtStream;

#define aclFloat16ToFloat(x) ((float)(x)
#define aclInit(x)
#define aclrtSetDevice(x)

#define aclrtCreateStream(x)

static inline void aclrtMallocHost(void**p, size_t sz){
    assert(sz);
    *p = malloc(sz);
}

#define aclrtMalloc(a,b,c) aclrtMallocHost(a,b)

#define aclrtMemcpy(dst, sz_dst, src, sz_src, type) std::memcpy(dst,src,sz_src)
#define aclrtSynchronizeStream(x)
#define aclrtFree(x) free(x)
#define aclrtFreeHost(x) free(x)
#define aclrtDestroyStream(x)
#define aclrtResetDevice(x)
#define aclFinalize(x)
#define set_flag(a,b,c)
#define wait_flag(a,b,c)

#endif