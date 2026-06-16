#ifndef _VQ_ACCEL_H_
#define _VQ_ACCEL_H_

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_vector.h>
#include <hls_stream.h>

// 固定输入 768x512 -> latent 128x192
#define NUM_Z 24576
#define TI    6
#define NUM_C 512
#define DIM   64

// 核心定点类型（保留不变）
typedef ap_fixed<18, 6>  calc_t;
typedef ap_fixed<42, 18> acc_t;
typedef hls::vector<float, 16> float16_v;

// ========== 重点修改：输出总线改为承载 索引 ==========
// 输入仍为 512bit(64*8bit int8 向量)，保持不变
typedef ap_uint<512> bus_t;
// 单个索引：0~511 → ap_uint<16> 完全够用
typedef ap_uint<16> idx_t;

extern "C" {
void vq_accel(
    bus_t *in_z,        // 输入：64维int8向量 (不变)
    bus_t *in_codebook, // 码本 (不变)
    idx_t *out_idx,     // 输出：码本索引 (替换原 out_z_q)
    float enc_scale,
    float dec_scale_inv
);
}

#endif
