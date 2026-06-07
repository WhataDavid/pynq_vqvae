//v16接口协议对齐,补充量化与反量化,结果正确20260513
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include "vq_accel.h"
//
//// 软件参考模型 (模拟 Jupyter Notebook 中的逻辑)
//void software_ref(int8_t *in, float *cb, int8_t *out, float esc, float dsc_inv) {
//    for (int n = 0; n < NUM_Z; n++) {
//        float z_e[DIM];
//        // 1. 反量化
//        for (int d = 0; d < DIM; d++) {
//            z_e[d] = (float)in[n * DIM + d] * esc;
//        }
//
//        // 2. 寻找最近向量
//        float min_dist = 1e38;
//        int best_idx = 0;
//        for (int c = 0; c < NUM_C; c++) {
//            float dist = 0;
//            for (int d = 0; d < DIM; d++) {
//                float diff = z_e[d] - cb[c * DIM + d];
//                dist += diff * diff;
//            }
//            if (dist < min_dist) {
//                min_dist = dist;
//                best_idx = c;
//            }
//        }
//
//        // 3. 量化并写回
//        for (int d = 0; d < DIM; d++) {
//            float val = cb[best_idx * DIM + d] * dsc_inv;
//            // 饱和截断逻辑
//            if (val > 127.0f) val = 127.0f;
//            if (val < -128.0f) val = -128.0f;
//            out[n * DIM + d] = (int8_t)std::round(val);
//        }
//    }
//}
//
//int main() {
//    std::cout << ">> 开始 V16 (int8<->float) 联合仿真验证..." << std::endl;
//
//    // 参数设置 (匹配 Jupyter 报告)
//    float enc_scale = 0.015625f;  // 2^-6
//    float dec_scale_inv = 32.0f;   // 2^5
//
//    // 分配内存
//    int8_t *in_z_raw = new int8_t[NUM_Z * DIM];
//    float  *cb_raw   = new float[NUM_C * DIM];
//    int8_t *out_hw   = new int8_t[NUM_Z * DIM];
//    int8_t *out_sw   = new int8_t[NUM_Z * DIM];
//
//    // 初始化模拟数据
//    for (int i = 0; i < NUM_Z * DIM; i++) in_z_raw[i] = (int8_t)((i % 255) - 128);
//    for (int i = 0; i < NUM_C * DIM; i++) cb_raw[i] = (float)(i % 10) * 0.1f;
//
//
//    // 转换指针类型供 HLS 使用，接口均为 512-bit 的 bus_t
//     bus_t *in_z_v    = reinterpret_cast<bus_t*>(in_z_raw);
//     bus_t *in_cb_v   = reinterpret_cast<bus_t*>(cb_raw);
//     bus_t *out_z_q_v = reinterpret_cast<bus_t*>(out_hw);
//
//    // 运行对比
//    std::cout << ">> 运行软件参考模型..." << std::endl;
//    software_ref(in_z_raw, cb_raw, out_sw, enc_scale, dec_scale_inv);
//
//    std::cout << ">> 运行 HLS 硬件仿真..." << std::endl;
//    vq_accel(in_z_v, in_cb_v, out_z_q_v, enc_scale, dec_scale_inv);
//
//    // 验证结果
//    int errors = 0;
//    for (int i = 0; i < NUM_Z * DIM; i++) {
//        // 允许 +/- 1 的量化误差 (round 策略差异)
//        if (std::abs(out_hw[i] - out_sw[i]) > 1) {
//            if (errors < 5) {
//                std::cout << "Error at [" << i << "]: HW=" << (int)out_hw[i]
//                          << ", SW=" << (int)out_sw[i] << std::endl;
//            }
//            errors++;
//        }
//    }
//
//    if (errors == 0) {
//        std::cout << "✅ TEST PASS! 硬件转换逻辑与软件完全一致。" << std::endl;
//    } else {
//        std::cout << "❌ TEST FAIL! 共有 " << errors << " 处误差。" << std::endl;
//    }
//
//    delete[] in_z_raw; delete[] cb_raw; delete[] out_hw; delete[] out_sw;
//    return (errors == 0) ? 0 : 1;
//}
//
//
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include "vq_accel.h"
//
//// 软件参考模型
//void software_ref(int8_t *in, float *cb, int8_t *out, float esc, float dsc_inv) {
//    for (int n = 0; n < NUM_Z; n++) {
//        float z_e[DIM];
//        for (int d = 0; d < DIM; d++) {
//            z_e[d] = (float)in[n * DIM + d] * esc;
//        }
//
//        float min_dist = 1e38;
//        int best_idx = 0;
//        for (int c = 0; c < NUM_C; c++) {
//            float dist = 0;
//            for (int d = 0; d < DIM; d++) {
//                float diff = z_e[d] - cb[c * DIM + d];
//                dist += diff * diff;
//            }
//            if (dist < min_dist) {
//                min_dist = dist;
//                best_idx = c;
//            }
//        }
//
//        for (int d = 0; d < DIM; d++) {
//            float val = cb[best_idx * DIM + d] * dsc_inv;
//            if (val > 127.0f) val = 127.0f;
//            if (val < -128.0f) val = -128.0f;
//            out[n * DIM + d] = (int8_t)std::round(val);
//        }
//    }
//}
//
//int main() {
//    std::cout << ">> 开始 V16 (Dual-Bundle + CB-Flag) 联合仿真验证..." << std::endl;
//
//    float enc_scale = 0.015625f;
//    float dec_scale_inv = 32.0f;
//
//    int8_t *in_z_raw = new int8_t[NUM_Z * DIM];
//    float  *cb_raw   = new float[NUM_C * DIM];
//    int8_t *out_hw   = new int8_t[NUM_Z * DIM];
//    int8_t *out_sw   = new int8_t[NUM_Z * DIM];
//
//    for (int i = 0; i < NUM_Z * DIM; i++) in_z_raw[i] = (int8_t)((i % 255) - 128);
//    for (int i = 0; i < NUM_C * DIM; i++) cb_raw[i] = (float)(i % 10) * 0.1f;
//
//    bus_t *in_z_v    = reinterpret_cast<bus_t*>(in_z_raw);
//    bus_t *in_cb_v   = reinterpret_cast<bus_t*>(cb_raw);
//    bus_t *out_z_q_v = reinterpret_cast<bus_t*>(out_hw);
//
//    std::cout << ">> 运行软件参考模型..." << std::endl;
//    software_ref(in_z_raw, cb_raw, out_sw, enc_scale, dec_scale_inv);
//
//    // 【关键修改点】
//    // 第一次运行：必须设置 load_cb_flag = 1 搬运码本
//    std::cout << ">> 运行 HLS 硬件仿真 (Frame 0, load_cb=1)..." << std::endl;
//    vq_accel(in_z_v, in_cb_v, out_z_q_v, enc_scale, dec_scale_inv, 1);
//
//    // 仿真第二次运行：设置 load_cb_flag = 0，验证 static cb_local 是否保持
//    std::cout << ">> 运行 HLS 硬件仿真 (Frame 1, load_cb=0)..." << std::endl;
//    vq_accel(in_z_v, in_cb_v, out_z_q_v, enc_scale, dec_scale_inv, 0);
//
//    // 验证结果
//    int errors = 0;
//    for (int i = 0; i < NUM_Z * DIM; i++) {
//        // 由于 HLS 使用了 ap_fixed 定点计算，允许 +/- 1 的微小舍入误差
//        if (std::abs(out_hw[i] - out_sw[i]) > 1) {
//            if (errors < 5) {
//                std::cout << "Error at [" << i << "]: HW=" << (int)out_hw[i]
//                          << ", SW=" << (int)out_sw[i] << std::endl;
//            }
//            errors++;
//        }
//    }
//
//    if (errors == 0) {
//        std::cout << "✅ TEST PASS! 硬件转换逻辑与软件完全一致。" << std::endl;
//    } else {
//        std::cout << "❌ TEST FAIL! 共有 " << errors << " 处误差。" << std::endl;
//        std::cout << "提示：如果是大规模误差，请检查码本加载逻辑；如果是极个别 +/-1，属于正常定点误差。" << std::endl;
//    }
//
//    delete[] in_z_raw; delete[] cb_raw; delete[] out_hw; delete[] out_sw;
//    return (errors == 0) ? 0 : 1;
//}

//#include <iostream>
//#include <vector>
//#include "vq_accel.h"
//
//int main() {
//    std::cout << ">> 开始 [跨帧数据更新] 稳定性验证..." << std::endl;
//
//    // --- 准备两组完全不同的输入数据 ---
//    int8_t *in_raw_1 = new int8_t[NUM_Z * DIM];
//    int8_t *in_raw_2 = new int8_t[NUM_Z * DIM];
//    float  *cb_raw   = new float[NUM_C * DIM];
//    int8_t *out_hw   = new int8_t[NUM_Z * DIM];
//
//    for (int i = 0; i < NUM_Z * DIM; i++) {
//        in_raw_1[i] = 10;  // 第一帧全为 10
//        in_raw_2[i] = 50;  // 第二帧全为 50
//    }
//    for (int i = 0; i < NUM_C * DIM; i++) cb_raw[i] = (float)(i % 10);
//
//    bus_t *cb_v = reinterpret_cast<bus_t*>(cb_raw);
//    bus_t *out_v = reinterpret_cast<bus_t*>(out_hw);
//
//    // --- 实验 A: 第一帧 (加载码本) ---
//    std::cout << ">> [Frame 0] 启动硬件 (load_cb=1, Input=10)..." << std::endl;
//    vq_accel(reinterpret_cast<bus_t*>(in_raw_1), cb_v, out_v, 1.0, 1.0, 1);
//    float mean1 = 0;
//    for(int i=0; i<100; i++) mean1 += out_hw[i];
//    std::cout << "   - Frame 0 Avg (First 100): " << mean1/100.0 << std::endl;
//
//    // --- 实验 B: 第二帧 (不加载码本，更换输入) ---
//    std::cout << ">> [Frame 1] 启动硬件 (load_cb=0, Input=50)..." << std::endl;
//    // 注意：这里传入 in_raw_2
//    vq_accel(reinterpret_cast<bus_t*>(in_raw_2), cb_v, out_v, 1.0, 1.0, 0);
//
//    float mean2 = 0;
//    for(int i=0; i<100; i++) mean2 += out_hw[i];
//    std::cout << "   - Frame 1 Avg (First 100): " << mean2/100.0 << std::endl;
//
//    // --- 判定逻辑 ---
//    if (mean1 == mean2) {
//        std::cout << "❌ 警告：硬件输出未随输入改变！数据锁死在第一帧。" << std::endl;
//    } else if (mean2 == 0 && mean1 != 0) {
//        std::cout << "❌ 警告：第二帧输出为 0，静态码本 (static URAM) 丢失。" << std::endl;
//    } else {
//        std::cout << "✅ 验证通过：硬件能够正确处理连续帧的数据更新。" << std::endl;
//    }
//
//    delete[] in_raw_1; delete[] in_raw_2; delete[] cb_raw; delete[] out_hw;
//    return 0;
//}

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include "vq_accel.h"

// ----------------------------------------
// 软件参考模型
// ----------------------------------------
void software_ref(int8_t *in, float *cb, int8_t *out, float esc, float dsc_inv) {
    for (int n = 0; n < NUM_Z; n++) {
        float z_e[DIM];

        // 1) 反量化
        for (int d = 0; d < DIM; d++) {
            z_e[d] = (float)in[n * DIM + d] * esc;
        }

        // 2) 找最近 codebook
        float min_dist = 1e38f;
        int best_idx = 0;

        for (int c = 0; c < NUM_C; c++) {
            float dist = 0.0f;

            for (int d = 0; d < DIM; d++) {
                float diff = z_e[d] - cb[c * DIM + d];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_idx = c;
            }
        }

        // 3) 量化写回
        for (int d = 0; d < DIM; d++) {
            float val = cb[best_idx * DIM + d] * dsc_inv;

            if (val > 127.0f)  val = 127.0f;
            if (val < -128.0f) val = -128.0f;

            out[n * DIM + d] = (int8_t)std::round(val);
        }
    }
}

int main() {
    std::cout << ">> 开始 768x512 VQ 联合仿真验证..." << std::endl;

    // 与板端一致
    float enc_scale = 0.015625f;   // 2^-6
    float dec_scale_inv = 32.0f;   // 1 / 0.03125

    // ----------------------------------------
    // 分配原始数据内存
    // ----------------------------------------
    int8_t *in_z_raw = new int8_t[NUM_Z * DIM];
    float  *cb_raw   = new float [NUM_C * DIM];
    int8_t *out_hw   = new int8_t[NUM_Z * DIM];
    int8_t *out_sw   = new int8_t[NUM_Z * DIM];

    // ----------------------------------------
    // 初始化测试数据
    // 这里只是仿真输入，不代表真实模型统计分布
    // ----------------------------------------
    for (int i = 0; i < NUM_Z * DIM; i++) {
        in_z_raw[i] = (int8_t)((i % 255) - 128);
    }

    for (int i = 0; i < NUM_C * DIM; i++) {
        cb_raw[i] = ((float)(i % 21) - 10.0f) * 0.05f;  // 大约 [-0.5, 0.5]
    }

    // ----------------------------------------
    // 512-bit 接口重解释
    // ----------------------------------------
    bus_t *in_z_v    = reinterpret_cast<bus_t*>(in_z_raw);
    bus_t *in_cb_v   = reinterpret_cast<bus_t*>(cb_raw);
    bus_t *out_z_q_v = reinterpret_cast<bus_t*>(out_hw);

    // ----------------------------------------
    // 跑软件参考
    // ----------------------------------------
    std::cout << ">> 运行软件参考模型..." << std::endl;
    software_ref(in_z_raw, cb_raw, out_sw, enc_scale, dec_scale_inv);

    // ----------------------------------------
    // 跑 HLS 硬件仿真
    // ----------------------------------------
    std::cout << ">> 运行 HLS 硬件仿真..." << std::endl;
    vq_accel(in_z_v, in_cb_v, out_z_q_v, enc_scale, dec_scale_inv);

    // ----------------------------------------
    // 对比结果
    // ----------------------------------------
    int errors = 0;
    for (int i = 0; i < NUM_Z * DIM; i++) {
        // 允许 ±1 量化误差
        if (std::abs((int)out_hw[i] - (int)out_sw[i]) > 1) {
            if (errors < 10) {
                std::cout << "Error at [" << i << "]: HW=" << (int)out_hw[i]
                          << ", SW=" << (int)out_sw[i] << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "✅ TEST PASS! 768x512 HLS 与软件参考一致。" << std::endl;
    } else {
        std::cout << "❌ TEST FAIL! 共有 " << errors << " 处误差。" << std::endl;
    }

    delete[] in_z_raw;
    delete[] cb_raw;
    delete[] out_hw;
    delete[] out_sw;

    return (errors == 0) ? 0 : 1;
}
