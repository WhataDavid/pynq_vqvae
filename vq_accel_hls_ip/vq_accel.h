//v15 512bit版本 latency=1.35E8
//#ifndef _VQ_ACCEL_H_
//#define _VQ_ACCEL_H_
//
//#include <ap_int.h>
//#include <hls_vector.h>
//
//// 参数定义
//#define NUM_Z 21875    // 125 * 175
//#define NUM_C 512      // Codebook 数量
//#define DIM   64       // 特征维度
//
//// 定义 512-bit 向量类型：一次读取 16 个 float
//typedef hls::vector<float, 16> float16_v;
//typedef float data_t;
//
//// 函数声明，使用 extern "C" 确保与 Vitis 链接一致
//extern "C" {
//    void vq_accel(
//        float16_v *in_z,         // [NUM_Z * DIM / 16]
//        float16_v *in_codebook,  // [NUM_C * DIM / 16]
//        float16_v *out_z_q       // [NUM_Z * DIM / 16]
//    );
//}
//
//#endif

//v24,修改接口,兼容xmodel，结果正确
//#ifndef _VQ_ACCEL_H_
//#define _VQ_ACCEL_H_
//
//#include <ap_int.h>
//#include <hls_vector.h>
//#include <hls_stream.h>
//
//// 常量定义
//#define NUM_Z 21875
//#define TI 7
//#define NUM_C 512
//#define DIM   64
//
//
//
//// 【核心类型】18位宽，完美贴合 DSP48 (27x18)
//typedef ap_fixed<18, 6> calc_t;
//typedef ap_fixed<42, 18> acc_t;
//
//// 内部计算类型：维持 V15 极致时序
//typedef hls::vector<float, 16> float16_v;
//
//// 接口类型：强制生成标准 512-bit AXI 接口，解决 PYNQ 全 0 问题
//typedef ap_uint<512> bus_t;
//
//extern "C" {
//    void vq_accel(
//        bus_t *in_z,
//        bus_t *in_codebook,
//        bus_t *out_z_q,
//        float enc_scale,
//        float dec_scale_inv
////		int   load_cb_flag
//    );
//}
//
//#endif

//改为768*512版本
#ifndef _VQ_ACCEL_H_
#define _VQ_ACCEL_H_

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_vector.h>
#include <hls_stream.h>

// ----------------------------------------
// 固定输入 768x512 -> latent 128x192
// ----------------------------------------
#define NUM_Z 24576
#define TI    8
#define NUM_C 512
#define DIM   64

// ----------------------------------------
// 核心类型
// ----------------------------------------
typedef ap_fixed<18, 6>  calc_t;
typedef ap_fixed<42, 18> acc_t;

// 保留定义（可不用）
typedef hls::vector<float, 16> float16_v;

// 512-bit AXI bus
typedef ap_uint<512> bus_t;

extern "C" {
void vq_accel(
    bus_t *in_z,
    bus_t *in_codebook,
    bus_t *out_z_q,
    float enc_scale,
    float dec_scale_inv
);
}

#endif
