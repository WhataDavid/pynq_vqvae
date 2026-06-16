#include "vq_accel_index.h"
#include <ap_fixed.h>
#include <stdint.h>

extern "C" {
void vq_accel(
    bus_t *in_z,
    bus_t *in_codebook,
    idx_t *out_idx,   // 改为索引输出
    float enc_scale,
    float dec_scale_inv
) {
    // ========== 接口约束更新 ==========
    #pragma HLS INTERFACE m_axi port=in_z         offset=slave bundle=gmem0 depth=24576
    #pragma HLS INTERFACE m_axi port=in_codebook  offset=slave bundle=gmem1 depth=2048
    // 输出索引 AXI，深度 = 总向量数 NUM_Z
    #pragma HLS INTERFACE m_axi port=out_idx      offset=slave bundle=gmem2 depth=24576
    #pragma HLS INTERFACE s_axilite port=in_z bundle=control
    #pragma HLS INTERFACE s_axilite port=in_codebook bundle=control
    #pragma HLS INTERFACE s_axilite port=out_idx bundle=control
    #pragma HLS INTERFACE s_axilite port=enc_scale bundle=control
    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // 量化参数（不变）
    calc_t fx_enc_scale = (calc_t)enc_scale;
    ap_fixed<16, 8> fx_dec_scale_inv = (ap_fixed<16, 8>)dec_scale_inv;

    // 本地码本（不变）
    static calc_t cb_local[NUM_C][DIM];
    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=URAM
    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete

    // 1) 加载 Codebook (逻辑完全不变)
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

    // 2) 主循环
    main_loop:
    for (int n = 0; n < NUM_Z; n += TI) {
        calc_t z_v[TI][DIM];
        #pragma HLS ARRAY_PARTITION variable=z_v complete dim=0

        // 2.1 读取输入 latent，反量化 (不变)
        read_z:
        for (int t = 0; t < TI; t++) {
            #pragma HLS PIPELINE II=1
            bus_t raw_in_bus = in_z[n + t];

            for (int k = 0; k < DIM; k++) {
                int8_t tmp_val = (int8_t)raw_in_bus.range(8 * k + 7, 8 * k);
                ap_fixed<10, 10> safe_i8 = tmp_val;
                z_v[t][k] = (calc_t)(safe_i8 * fx_enc_scale);
            }
        }

        // 2.2 初始化最小距离、最优索引 (不变)
        acc_t min_dist[TI];
        int   best_idx[TI];
        #pragma HLS ARRAY_PARTITION variable=min_dist complete
        #pragma HLS ARRAY_PARTITION variable=best_idx complete

        init_min:
        for (int t = 0; t < TI; t++) {
            #pragma HLS UNROLL
            min_dist[t] = (acc_t)100000.0;
            best_idx[t] = 0;
        }

        // 2.3 遍历码本找最近邻 (距离计算逻辑完全不变)
        find_min:
        for (int c = 0; c < NUM_C; c++) {
            #pragma HLS PIPELINE II=1 rewind

            acc_t dist_accum[TI];
            #pragma HLS ARRAY_PARTITION variable=dist_accum complete

            init_dist:
            for (int t = 0; t < TI; t++) {
                #pragma HLS UNROLL
                dist_accum[t] = 0;
            }

            calc_dist:
            for (int j = 0; j < 4; j++) {
                for (int t = 0; t < TI; t++) {
                    acc_t partial_sum = 0;
                    #pragma HLS EXPRESSION_BALANCE
                    for (int v = 0; v < 16; v++) {
                        int k = j * 16 + v;
                        calc_t diff = z_v[t][k] - cb_local[c][k];
                        acc_t sq = diff * diff;
                        #pragma HLS BIND_OP variable=sq op=mul impl=dsp
                        partial_sum += sq;
                    }
                    dist_accum[t] += partial_sum;
                }
            }

            update_min:
            for (int t = 0; t < TI; t++) {
                if (dist_accum[t] < min_dist[t]) {
                    min_dist[t] = dist_accum[t];
                    best_idx[t] = c;
                }
            }
        }

        // ===================== 重点修改 =====================
        // 2.4 直接输出 索引，不再重构向量
        write_idx:
        for (int t = 0; t < TI; t++) {
            #pragma HLS PIPELINE II=1
            // best_idx 0~511 → 转为 idx_t(ap_uint<16>) 写入AXI
            out_idx[n + t] = (idx_t)best_idx[t];
        }
        // ====================================================
    }
}
}
