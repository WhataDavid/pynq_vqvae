#include "vq_dequant.h"
#include <stdint.h>

extern "C" {
void vq_dequant(
    idx_t        *in_idx,
    bus_t        *in_codebook,
    bus_t        *out_feature,
    float         dec_scale_inv
) {
    #pragma HLS INTERFACE m_axi port=in_idx         offset=slave bundle=gmem0 depth=24576
    #pragma HLS INTERFACE m_axi port=in_codebook    offset=slave bundle=gmem1 depth=2048
    #pragma HLS INTERFACE m_axi port=out_feature    offset=slave bundle=gmem2 depth=24576
    #pragma HLS INTERFACE s_axilite port=in_idx        bundle=control
    #pragma HLS INTERFACE s_axilite port=in_codebook  bundle=control
    #pragma HLS INTERFACE s_axilite port=out_feature  bundle=control
    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
    #pragma HLS INTERFACE s_axilite port=return       bundle=control

    // 定点缩放系数，和原工程对齐
    ap_fixed<16, 8> fx_dec_scale_inv = (ap_fixed<16, 8>)dec_scale_inv;

    // 本地码本缓存，存储/分区策略和 vq_accel 完全一致
    static calc_t cb_local[NUM_C][DIM];
    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete

    // ==============================================
    // 1. 加载码本（逻辑 1:1 复刻你 vq_accel）
    // 每个 codeword 64 float = 4 * 512bit
    // ==============================================
    load_cb:
    for (int i = 0; i < NUM_C; i++) {
        for (int j = 0; j < 4; j++) {
            #pragma HLS PIPELINE II=1
            bus_t raw_cb = in_codebook[i * 4 + j];

            for (int v = 0; v < 16; v++) {
                uint32_t bits = (uint32_t)raw_cb.range(32 * v + 31, 32 * v);
                float f_val = *(float*)(&bits);
                cb_local[i][j * 16 + v] = (calc_t)f_val;
            }
        }
    }

    // ==============================================
    // 2. 主流程：索引查表 → 缩放 → 饱和 → 打包输出
    // ==============================================
    main_loop:
    for (int n = 0; n < NUM_Z; n += TI) {
        #pragma HLS PIPELINE II=1

        // 并行读取 TI 个索引
        idx_t curr_idx[TI];
        #pragma HLS ARRAY_PARTITION variable=curr_idx complete
        for(int t = 0; t < TI; t++){
            curr_idx[t] = in_idx[n + t];
        }

        // 查表+转换+打包输出
        write_out:
        for (int t = 0; t < TI; t++) {
            #pragma HLS PIPELINE II=1
            bus_t raw_out_bus = 0;
            int select_idx = curr_idx[t];

            convert_out:
            for (int k = 0; k < DIM; k++) {
                // 码本取值 + 缩放
                ap_fixed<24, 12> val_scaled = cb_local[select_idx][k] * fx_dec_scale_inv;
                // 饱和截断到 int8
                ap_fixed<8, 8, AP_RND, AP_SAT> sat_val = val_scaled;
                // 8bit 打包进 512bit 总线
                raw_out_bus.range(8 * k + 7, 8 * k) = (ap_uint<8>)sat_val;
            }
            out_feature[n + t] = raw_out_bus;
        }
    }
}
}
