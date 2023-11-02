#pragma once

#include "CompileFlags.h"

#if COMPILECUDA == 1

#include "mGPU.h"

//multi-GPU configuration
extern mGPUConfig mGPU;

#endif
