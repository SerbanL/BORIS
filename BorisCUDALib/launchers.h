#pragma once

//set number of cuda threads per block
#define CUDATHREADS	128

//should not launch a kernel with less than this number of threads per block, otherwise very inefficient (warp size)
#define MIN_CUDATHREADS 32

//maximum allowed number of cuda threads per block
#define MAX_CUDATHREADS	1024

//maximum number of cuda threads used for block reductions
#define MAX_BLOCKREDUCTION_CUDATHREADS	512
