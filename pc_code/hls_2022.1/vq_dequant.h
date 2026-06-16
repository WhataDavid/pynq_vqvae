#ifndef _VQ_DEQUANT_H_
#define _VQ_DEQUANT_H_

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_vector.h>
#include <hls_stream.h>

// 与 vq_accel 完全一致宏定义
#define NUM_Z 24576
#define TI    8
#define NUM_C 512
#define DIM   64

// 类型对齐
typedef ap_fixed<18, 6>  calc_t;
typedef ap_fixed<42, 18> acc_t;

// 512bit AXI 总线
typedef ap_uint<512> bus_t;
// 索引类型
typedef ap_uint<16> idx_t;

extern "C" {
void vq_dequant(
    idx_t        *in_idx,
    bus_t        *in_codebook,
    bus_t        *out_feature,
    float         dec_scale_inv
);
}

#endif
