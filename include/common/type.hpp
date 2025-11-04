#ifndef _INCLUDE_DAVINCI_TYPE_H_
#define _INCLUDE_DAVINCI_TYPE_H_
#if defined(__DAV_V220) || defined(__DAV_V310)
#define __aicore__ [aicore]
#else
#define __aicore__
#endif
#define PTO_INLINE inline __attribute__((always_inline))

#define __PTO_INSTR__ __aicore__ PTO_INLINE
#endif