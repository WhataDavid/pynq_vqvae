//v24,修改接口,兼容xmodel，结果正确，latency=206ms
//#include "vq_accel.h"
//
//extern "C" {
//void vq_accel(
//	    bus_t *in_z,
//	    bus_t *in_codebook,
//	    bus_t *out_z_q,
//	    float enc_scale,
//	    float dec_scale_inv
//	) {
//	    // 分开接口 Bundle 解决死锁
//	    #pragma HLS INTERFACE m_axi port=in_z         offset=slave bundle=gmem0 depth=21875
//	    #pragma HLS INTERFACE m_axi port=in_codebook  offset=slave bundle=gmem1 depth=2048
//	    #pragma HLS INTERFACE m_axi port=out_z_q      offset=slave bundle=gmem2 depth=21875
//
//	    #pragma HLS INTERFACE s_axilite port=in_z          bundle=control
//	    #pragma HLS INTERFACE s_axilite port=in_codebook   bundle=control
//	    #pragma HLS INTERFACE s_axilite port=out_z_q       bundle=control
//	    #pragma HLS INTERFACE s_axilite port=enc_scale     bundle=control
//	    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
//	    #pragma HLS INTERFACE s_axilite port=return        bundle=control
//
//    static float16_v cb_local[512][4];
//    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=URAM
////#pragma HLS ARRAY_PARTITION variable=cb_local dim=2 cyclic factor=4
//    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete
//
//    // 1. 预加载 Codebook (最稳健的位块加载方式)
//    load_cb: for(int i=0; i<512; i++) {
//        for(int j=0; j<4; j++) {
////            #pragma HLS PIPELINE II=1
//            bus_t raw_cb = in_codebook[i * 4 + j];
//            float16_v tmp_v;
//            for(int v=0; v<16; v++) {
////                #pragma HLS UNROLL
//                uint32_t bits = raw_cb.range(32*v+31, 32*v);
//                tmp_v[v] = *(float*)(&bits);
//            }
//            cb_local[i][j] = tmp_v;
//        }
//    }
//
//    // 2. 极致主循环
//    main_loop: for(int n=0; n<21875; n++) {
//        float16_v z_v[4];
//        #pragma HLS ARRAY_PARTITION variable=z_v complete
//
//        // 加载输入并立即进行位转换加载
//        bus_t raw_in_bus = in_z[n];
//        for (int i = 0; i < 4; i++) {
//            #pragma HLS UNROLL
//            for (int v = 0; v < 16; v++) {
//                #pragma HLS UNROLL
//                // 显式提取 int8 并应用 enc_scale
//                int8_t tmp_val = (int8_t)raw_in_bus.range(8*(i*16+v)+7, 8*(i*16+v));
//                z_v[i][v] = (float)tmp_val * enc_scale;
//            }
//        }
//
//        float min_dist = 1e38f;
//        int best_idx = 0;
//
//        // 3. VQ 核心计算 (保持你的 II=1 rewind 极致时序)
//        find_min: for(int c=0; c<512; c++) {
//            #pragma HLS PIPELINE II=1 rewind
//            float dist_accum = 0.0f;
//
//            calc_dist: for(int j=0; j<4; j++) {
//                #pragma HLS UNROLL
//                float16_v diff = z_v[j] - cb_local[c][j];
//                float16_v sq = diff * diff;
//
//                float partial_sum = 0.0f;
//                for(int v=0; v<16; v++) {
//                    #pragma HLS UNROLL
//                    partial_sum += sq[v];
//                }
//                dist_accum += partial_sum;
//            }
//
//            if(dist_accum < min_dist) {
//                min_dist = dist_accum;
//                best_idx = c; // 修正后的变量
//            }
//        }
//
//        // 4. 写回转换
//        bus_t raw_out_bus = 0;
//        convert_out: for (int i = 0; i < 4; i++) {
//            #pragma HLS UNROLL
//            for (int v = 0; v < 16; v++) {
//                #pragma HLS UNROLL
//                float val = cb_local[best_idx][i][v] * dec_scale_inv;
//                int8_t out_i8;
//                if (val > 127.0f) out_i8 = 127;
//                else if (val < -128.0f) out_i8 = -128;
//                else out_i8 = (int8_t)val;
//
//                raw_out_bus.range(8*(i*16+v)+7, 8*(i*16+v)) = out_i8;
//            }
//        }
//        out_z_q[n] = raw_out_bus;
//    }
//}
//}

//v24基础上,尝试优化时序，结果正确，还是206ms
//#include "vq_accel.h"
//#include <hls_stream.h>
//
//// 将数组定义为全局静态，物理分区才能在所有进程中生效
//static float16_v cb_local[512][4];
//
//// 1. 初始化模块：预加载 Codebook
//void init_codebook(bus_t *in_codebook) {
//    load_cb: for(int i=0; i<512; i++) {
//        for(int j=0; j<4; j++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_cb = in_codebook[i * 4 + j];
//            float16_v tmp_v;
//            for(int v=0; v<16; v++) {
//                #pragma HLS UNROLL
//                uint32_t bits = raw_cb.range(32*v+31, 32*v);
//                tmp_v[v] = *(float*)(&bits);
//            }
//            cb_local[i][j] = tmp_v;
//        }
//    }
//}
//
//// 2. 读取模块
//void load_input(bus_t *in, hls::stream<bus_t>& in_stream) {
//    for (int n = 0; n < NUM_Z; n++) {
//        #pragma HLS PIPELINE II=1
//        in_stream.write(in[n]);
//    }
//}
//
//// 3. 计算模块
//void compute_vq(
//    hls::stream<bus_t>& in_stream,
//    hls::stream<bus_t>& out_stream,
//    float enc_scale,
//    float dec_scale_inv
//) {
//    // 强制声明分区，确保计算单元能看到 4 个 URAM 端口
//    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete
//
//    main_compute: for (int n = 0; n < NUM_Z; n++) {
//        bus_t raw_in_bus = in_stream.read();
//
//        float16_v z_v[4];
//        #pragma HLS ARRAY_PARTITION variable=z_v complete
//
//        // 转换输入
//        for (int i = 0; i < 4; i++) {
//            #pragma HLS UNROLL
//            for (int v = 0; v < 16; v++) {
//                #pragma HLS UNROLL
//                int8_t tmp_val = (int8_t)raw_in_bus.range(8*(i*16+v)+7, 8*(i*16+v));
//                z_v[i][v] = (float)tmp_val * enc_scale;
//            }
//        }
//
//        float min_dist = 1e38f;
//        int best_idx = 0;
//
//        // VQ 核心寻找最近邻 (目标 II=1)
//        find_min: for (int c = 0; c < 512; c++) {
//            #pragma HLS PIPELINE II=1 rewind
//            float dist_accum = 0.0f;
//            calc_dist: for (int j = 0; j < 4; j++) {
//                #pragma HLS UNROLL
//                float16_v diff = z_v[j] - cb_local[c][j];
//                float16_v sq = diff * diff;
//                for (int v = 0; v < 16; v++) {
//                    #pragma HLS UNROLL
//                    dist_accum += sq[v];
//                }
//            }
//            if (dist_accum < min_dist) {
//                min_dist = dist_accum;
//                best_idx = c;
//            }
//        }
//
//        // 转换输出
//        bus_t raw_out_bus = 0;
//        convert_out: for (int i = 0; i < 4; i++) {
//            #pragma HLS UNROLL
//            for (int v = 0; v < 16; v++) {
//                #pragma HLS UNROLL
//                float val = cb_local[best_idx][i][v] * dec_scale_inv;
//                int8_t out_i8 = (val > 127.0f) ? 127 : (val < -128.0f) ? -128 : (int8_t)val;
//                raw_out_bus.range(8*(i*16+v)+7, 8*(i*16+v)) = out_i8;
//            }
//        }
//        out_stream.write(raw_out_bus);
//    }
//}
//
//// 4. 存储模块
//void store_output(bus_t *out, hls::stream<bus_t>& out_stream) {
//    for (int n = 0; n < NUM_Z; n++) {
//        #pragma HLS PIPELINE II=1
//        out[n] = out_stream.read();
//    }
//}
//
//extern "C" {
//void vq_accel(
//    bus_t *in_z,
//    bus_t *in_codebook,
//    bus_t *out_z_q,
//    float enc_scale,
//    float dec_scale_inv
//) {
//    #pragma HLS INTERFACE m_axi port=in_z offset=slave bundle=gmem0 depth=21875 max_read_burst_length=256
//    #pragma HLS INTERFACE m_axi port=in_codebook offset=slave bundle=gmem1 depth=2048
//    #pragma HLS INTERFACE m_axi port=out_z_q offset=slave bundle=gmem2 depth=21875 max_write_burst_length=256
//
//    #pragma HLS INTERFACE s_axilite port=in_z bundle=control
//    #pragma HLS INTERFACE s_axilite port=in_codebook bundle=control
//    #pragma HLS INTERFACE s_axilite port=out_z_q bundle=control
//    #pragma HLS INTERFACE s_axilite port=enc_scale bundle=control
//    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
//    #pragma HLS INTERFACE s_axilite port=return bundle=control
//
//    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=URAM
//    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete
//
//    // 1. 先执行初始化 (Sequential 执行)
//    init_codebook(in_codebook);
//
//    // 2. 准备并行流水线
//    hls::stream<bus_t> in_stream("in_stream");
//    hls::stream<bus_t> out_stream("out_stream");
//    #pragma HLS STREAM variable=in_stream depth=32
//    #pragma HLS STREAM variable=out_stream depth=32
//
//    // 3. 开启 Dataflow (只包含 Load/Compute/Store 子任务)
//    #pragma HLS DATAFLOW
//    load_input(in_z, in_stream);
//    compute_vq(in_stream, out_stream, enc_scale, dec_scale_inv);
//    store_output(out_z_q, out_stream);
//}
//}

//参考刘谦的并行设计，布局拥塞，无法编译比特流
//#include "vq_accel.h"
//#include <hls_stream.h>
//
//// 增加并行处理块的大小 (Ti)。21875 能被 5 整除 (21875 / 5 = 4375)
//#define TI 5
//
//// 将数组定义为全局静态，物理分区才能在所有进程中生效
//static float16_v cb_local[512][4];
//
//// 1. 初始化模块：预加载 Codebook
//void init_codebook(bus_t *in_codebook) {
//    load_cb: for(int i=0; i<512; i++) {
//        for(int j=0; j<4; j++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_cb = in_codebook[i * 4 + j];
//            float16_v tmp_v;
//            for(int v=0; v<16; v++) {
//                #pragma HLS UNROLL
//                uint32_t bits = raw_cb.range(32*v+31, 32*v);
//                tmp_v[v] = *(float*)(&bits);
//            }
//            cb_local[i][j] = tmp_v;
//        }
//    }
//}
//
//// 2. 读取模块
//void load_input(bus_t *in, hls::stream<bus_t>& in_stream) {
//    for (int n = 0; n < NUM_Z; n++) {
//        #pragma HLS PIPELINE II=1
//        in_stream.write(in[n]);
//    }
//}
//
//// 3. 计算模块
//void compute_vq(
//    hls::stream<bus_t>& in_stream,
//    hls::stream<bus_t>& out_stream,
//    float enc_scale,
//    float dec_scale_inv
//) {
//    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete
//
//    // 每次处理 TI 个输入向量，减少遍历 codebook 的次数
//    main_compute: for (int n = 0; n < NUM_Z; n += TI) {
//
//        float16_v z_v[TI][4];
//        #pragma HLS ARRAY_PARTITION variable=z_v complete
//
//        // 读取 TI 个输入并转换
//        read_z: for (int t = 0; t < TI; t++) {
//            bus_t raw_in_bus = in_stream.read();
//            for (int i = 0; i < 4; i++) {
//                #pragma HLS UNROLL
//                for (int v = 0; v < 16; v++) {
//                    #pragma HLS UNROLL
//                    int8_t tmp_val = (int8_t)raw_in_bus.range(8*(i*16+v)+7, 8*(i*16+v));
//                    z_v[t][i][v] = (float)tmp_val * enc_scale;
//                }
//            }
//        }
//
//        float min_dist[TI];
//        int best_idx[TI];
//        #pragma HLS ARRAY_PARTITION variable=min_dist complete
//        #pragma HLS ARRAY_PARTITION variable=best_idx complete
//
//        for (int t = 0; t < TI; t++) {
//            #pragma HLS UNROLL
//            min_dist[t] = 1e38f;
//            best_idx[t] = 0;
//        }
//
//        // 核心：复用 codebook 数据，并行计算 TI 个向量的距离
//        find_min: for (int c = 0; c < 512; c++) {
//            #pragma HLS PIPELINE II=1 rewind
//
//            float dist_accum[TI];
//            #pragma HLS ARRAY_PARTITION variable=dist_accum complete
//            for (int t = 0; t < TI; t++) {
//                #pragma HLS UNROLL
//                dist_accum[t] = 0.0f;
//            }
//
//            calc_dist: for (int j = 0; j < 4; j++) {
//                #pragma HLS UNROLL
//                float16_v cb_cache = cb_local[c][j]; // 提前读取缓存，减少端口压力
//
//                for (int t = 0; t < TI; t++) {
//                    #pragma HLS UNROLL
//                    float16_v diff = z_v[t][j] - cb_cache;
//                    float16_v sq = diff * diff;
//                    // 强制开启表达式平衡，构建高效的浮点加法树
//                    #pragma HLS EXPRESSION_BALANCE
//                    for (int v = 0; v < 16; v++) {
//                        #pragma HLS UNROLL
//                        dist_accum[t] += sq[v];
//                    }
//                }
//            }
//
//            // 并行更新最小值
//            for (int t = 0; t < TI; t++) {
//                #pragma HLS UNROLL
//                if (dist_accum[t] < min_dist[t]) {
//                    min_dist[t] = dist_accum[t];
//                    best_idx[t] = c;
//                }
//            }
//        }
//
//        // 转换并输出 TI 个结果
//        write_out: for (int t = 0; t < TI; t++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_out_bus = 0;
//            convert_out: for (int i = 0; i < 4; i++) {
//                #pragma HLS UNROLL
//                for (int v = 0; v < 16; v++) {
//                    #pragma HLS UNROLL
//                    float val = cb_local[best_idx[t]][i][v] * dec_scale_inv;
//                    int8_t out_i8 = (val > 127.0f) ? 127 : (val < -128.0f) ? -128 : (int8_t)val;
//                    raw_out_bus.range(8*(i*16+v)+7, 8*(i*16+v)) = out_i8;
//                }
//            }
//            out_stream.write(raw_out_bus);
//        }
//    }
//}
//
//// 4. 存储模块
//void store_output(bus_t *out, hls::stream<bus_t>& out_stream) {
//    for (int n = 0; n < NUM_Z; n++) {
//        #pragma HLS PIPELINE II=1
//        out[n] = out_stream.read();
//    }
//}
//
//extern "C" {
//void vq_accel(
//    bus_t *in_z,
//    bus_t *in_codebook,
//    bus_t *out_z_q,
//    float enc_scale,
//    float dec_scale_inv
//) {
//    #pragma HLS INTERFACE m_axi port=in_z offset=slave bundle=gmem0 depth=21875 max_read_burst_length=256
//    #pragma HLS INTERFACE m_axi port=in_codebook offset=slave bundle=gmem1 depth=2048
//    #pragma HLS INTERFACE m_axi port=out_z_q offset=slave bundle=gmem2 depth=21875 max_write_burst_length=256
//
//    #pragma HLS INTERFACE s_axilite port=in_z bundle=control
//    #pragma HLS INTERFACE s_axilite port=in_codebook bundle=control
//    #pragma HLS INTERFACE s_axilite port=out_z_q bundle=control
//    #pragma HLS INTERFACE s_axilite port=enc_scale bundle=control
//    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
//    #pragma HLS INTERFACE s_axilite port=return bundle=control
//
//    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=URAM
//    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete
//
//    init_codebook(in_codebook);
//
//    hls::stream<bus_t> in_stream("in_stream");
//    hls::stream<bus_t> out_stream("out_stream");
//    // 放大 FIFO 以防止数据流阻塞
//    #pragma HLS STREAM variable=in_stream depth=64
//    #pragma HLS STREAM variable=out_stream depth=64
//
//    #pragma HLS DATAFLOW
//    load_input(in_z, in_stream);
//    compute_vq(in_stream, out_stream, enc_scale, dec_scale_inv);
//    store_output(out_z_q, out_stream);
//}
//}

//参考刘谦的并行设计，尝试处理拥塞，34e7ms，但无法布局布线
//#include "vq_accel.h"
//
//#define TI 5
//
//extern "C" {
//void vq_accel(
//    bus_t *in_z,
//    bus_t *in_codebook,
//    bus_t *out_z_q,
//    float enc_scale,
//    float dec_scale_inv
//) {
//    // 分开接口 Bundle 解决死锁，保留原版的稳定 AXI 配置
//    #pragma HLS INTERFACE m_axi port=in_z         offset=slave bundle=gmem0 depth=21875
//    #pragma HLS INTERFACE m_axi port=in_codebook  offset=slave bundle=gmem1 depth=2048
//    #pragma HLS INTERFACE m_axi port=out_z_q      offset=slave bundle=gmem2 depth=21875
//
//    #pragma HLS INTERFACE s_axilite port=in_z          bundle=control
//    #pragma HLS INTERFACE s_axilite port=in_codebook   bundle=control
//    #pragma HLS INTERFACE s_axilite port=out_z_q       bundle=control
//    #pragma HLS INTERFACE s_axilite port=enc_scale     bundle=control
//    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
//    #pragma HLS INTERFACE s_axilite port=return        bundle=control
//
//    static float16_v cb_local[512][4];
//    // 原版中能跑通，就继续保留 URAM（如果再次报资源不足可随时改为 BRAM）
//    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=URAM
//    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete
//
//    // 1. 预加载 Codebook (最稳健的位块加载方式)
//    load_cb: for(int i=0; i<512; i++) {
//        for(int j=0; j<4; j++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_cb = in_codebook[i * 4 + j];
//            float16_v tmp_v;
//            for(int v=0; v<16; v++) {
//                #pragma HLS UNROLL
//                uint32_t bits = raw_cb.range(32*v+31, 32*v);
//                tmp_v[v] = *(float*)(&bits);
//            }
//            cb_local[i][j] = tmp_v;
//        }
//    }
//
//    // 2. 极致主循环 (按 TI 步进)
//    main_loop: for(int n=0; n<21875; n+=TI) {
//
//        float16_v z_v[TI][4];
//        #pragma HLS ARRAY_PARTITION variable=z_v complete
//
//        // 连续加载 TI 个输入并立即进行位转换
//        read_z: for(int t=0; t<TI; t++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_in_bus = in_z[n + t];
//            for (int i = 0; i < 4; i++) {
//                #pragma HLS UNROLL
//                for (int v = 0; v < 16; v++) {
//                    #pragma HLS UNROLL
//                    int8_t tmp_val = (int8_t)raw_in_bus.range(8*(i*16+v)+7, 8*(i*16+v));
//                    z_v[t][i][v] = (float)tmp_val * enc_scale;
//                }
//            }
//        }
//
//        float min_dist[TI];
//        int best_idx[TI];
//        #pragma HLS ARRAY_PARTITION variable=min_dist complete
//        #pragma HLS ARRAY_PARTITION variable=best_idx complete
//
//        for(int t=0; t<TI; t++) {
//            #pragma HLS UNROLL
//            min_dist[t] = 1e38f;
//            best_idx[t] = 0;
//        }
//
//        // 3. VQ 核心计算：一次性并行计算 TI 个向量的距离
//        find_min: for(int c=0; c<512; c++) {
//            #pragma HLS PIPELINE II=1 rewind
//
//            float dist_accum[TI];
//            #pragma HLS ARRAY_PARTITION variable=dist_accum complete
//            for(int t=0; t<TI; t++) {
//                #pragma HLS UNROLL
//                dist_accum[t] = 0.0f;
//            }
//
//            calc_dist: for(int j=0; j<4; j++) {
////                #pragma HLS UNROLL
//#pragma HLS PIPELINE II=1
//                float16_v cb_cache = cb_local[c][j]; // 提前读取，复用端口
//
//                for(int t=0; t<TI; t++) {
////                    #pragma HLS UNROLL
//#pragma HLS PIPELINE II=1
//                    float16_v diff = z_v[t][j] - cb_cache;
//                    float16_v sq = diff * diff;
//
//                    float partial_sum = 0.0f;
//                    // 强制平衡加法树，防止 TI=5 展开后导致时序爆炸
//                    #pragma HLS EXPRESSION_BALANCE
//                    for(int v=0; v<16; v++) {
////                        #pragma HLS UNROLL
//#pragma HLS PIPELINE II=1
//                        partial_sum += sq[v];
//                    }
//                    dist_accum[t] += partial_sum;
//                }
//            }
//
//            // 更新 TI 个最小值
//            for(int t=0; t<TI; t++) {
//                #pragma HLS UNROLL
//                if(dist_accum[t] < min_dist[t]) {
//                    min_dist[t] = dist_accum[t];
//                    best_idx[t] = c;
//                }
//            }
//        }
//
//        // 4. 写回转换 (依次输出 TI 个结果)
//        write_out: for(int t=0; t<TI; t++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_out_bus = 0;
//            convert_out: for (int i = 0; i < 4; i++) {
////                #pragma HLS UNROLL
//#pragma HLS PIPELINE II=1
//                for (int v = 0; v < 16; v++) {
////                    #pragma HLS UNROLL
//#pragma HLS PIPELINE II=1
//                    float val = cb_local[best_idx[t]][i][v] * dec_scale_inv;
//                    int8_t out_i8;
//                    if (val > 127.0f) out_i8 = 127;
//                    else if (val < -128.0f) out_i8 = -128;
//                    else out_i8 = (int8_t)val;
//
//                    raw_out_bus.range(8*(i*16+v)+7, 8*(i*16+v)) = out_i8;
//                }
//            }
//            out_z_q[n + t] = raw_out_bus;
//        }
//    }
//}
//}

//尽量使用刘谦代码，减少资源使用32ms,无法布线
//#include "vq_accel.h"
//#include <ap_fixed.h>
//
//// 1. 完全复用 vq.h 中的高性能类型定义
//typedef ap_fixed<16, 7, AP_RND, AP_SAT> data_t;
//
//// 定义原 IP 中的分块参数（直接从你的 vq.h 复制）
//#define Tp 16
//#define Tn 1
//#define Tm 16
//#define Ti 4
//#define MAX_LEN 1024
//
//// --- 声明你仓库中已有的高性能子函数 ---
//// 这些函数直接从你的 vq.cpp 逻辑移植过来，不做任何修改，确保面积最小
//void compute_core(data_t fm_in_buff[Ti][Tp], data_t fm_out_buff[Ti][Tm][Tp], data_t cb_buff[Tm][Tn]) {
//    #pragma HLS INLINE
//    for(int i=0; i<Tp; i++){
//        for(int t=0; t<Ti; t+=2) {
//            #pragma HLS PIPELINE II=1
//            for(int mm=0; mm<Tm; mm++){
//                #pragma HLS UNROLL
//                data_t diff1 = fm_in_buff[t][i] - cb_buff[mm][0];
//                fm_out_buff[t][mm][i] += diff1 * diff1;
//                data_t diff2 = fm_in_buff[t+1][i] - cb_buff[mm][0];
//                fm_out_buff[t+1][mm][i] += diff2 * diff2;
//            }
//        }
//    }
//}
//
//extern "C" {
//void vq_accel(
//    bus_t *in_z,         // 512-bit 接口
//    bus_t *in_codebook,  // 512-bit 接口
//    bus_t *out_z_q,      // 512-bit 接口
//    float enc_scale,
//    float dec_scale_inv
//) {
//    // 接口定义保持不变，兼容 Jupyter
//    #pragma HLS INTERFACE m_axi port=in_z         offset=slave bundle=gmem0 depth=21875
//    #pragma HLS INTERFACE m_axi port=in_codebook  offset=slave bundle=gmem1 depth=2048
//    #pragma HLS INTERFACE m_axi port=out_z_q      offset=slave bundle=gmem2 depth=21875
//    #pragma HLS INTERFACE s_axilite port=in_z bundle=control
//    #pragma HLS INTERFACE s_axilite port=in_codebook bundle=control
//    #pragma HLS INTERFACE s_axilite port=out_z_q bundle=control
//    #pragma HLS INTERFACE s_axilite port=enc_scale bundle=control
//    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
//    #pragma HLS INTERFACE s_axilite port=return bundle=control
//
//    // 内部存储：完全复用你的静态 Buffer 设计
//    static data_t cb_local[512][64];
//    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=BRAM
//
//    // 1. 高效加载 Codebook (来自 vq_lookup 的 load_codebook 逻辑)
//    // 直接将浮点 Codebook 转换为定点存储在片上
//    for(int i=0; i<512; i++) {
//        for(int j=0; j<4; j++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw = in_codebook[i*4 + j];
//            for(int v=0; v<16; v++) {
//                uint32_t bits = raw.range(32*v+31, 32*v);
//                float f_val = *(float*)(&bits);
//                cb_local[i][j*16 + v] = (data_t)f_val;
//            }
//        }
//    }
//
//    // 2. 主循环：采用 vq.cpp 的 Tiling 思想，但针对 512-bit 接口做了适配
//    // 每次处理 4 个向量 (Ti=4)，每个向量 64 维
//    main_loop: for(int n=0; n<21875; n+=Ti) {
//
//        data_t fm_in_buff[Ti][64];
//        #pragma HLS ARRAY_PARTITION variable=fm_in_buff complete dim=1
//
//        data_t dist_min[Ti];
//        unsigned short best_idx[Ti];
//        #pragma HLS ARRAY_PARTITION variable=dist_min complete
//        #pragma HLS ARRAY_PARTITION variable=best_idx complete
//
//        // 加载输入 (类似于你的 load_input)
//        for(int t=0; t<Ti; t++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw = in_z[n + t];
//            for(int k=0; k<64; k++) {
//                int8_t val = (int8_t)raw.range(8*k+7, 8*k);
//                fm_in_buff[t][k] = (data_t)((float)val * enc_scale);
//            }
//        }
//
//        // 初始化最小值
//        for(int t=0; t<Ti; t++) { dist_min[t] = 127.0; best_idx[t] = 0; }
//
//        // 3. 核心 VQ 搜索：这是你 vq.cpp 中最省面积的部分
//        search_loop: for(int c=0; c<512; c++) {
//            #pragma HLS PIPELINE II=1
//            data_t acc[Ti] = {0,0,0,0};
//            #pragma HLS ARRAY_PARTITION variable=acc complete
//
//            for(int k=0; k<64; k++) {
//                #pragma HLS UNROLL
//                data_t cb_val = cb_local[c][k];
//                for(int t=0; t<Ti; t++) {
//                    data_t diff = fm_in_buff[t][k] - cb_val;
//                    acc[t] += diff * diff;
//                }
//            }
//
//            for(int t=0; t<Ti; t++) {
//                if(acc[t] < dist_min[t]) {
//                    dist_min[t] = acc[t];
//                    best_idx[t] = c;
//                }
//            }
//        }
//
//        // 4. 写回转换 (整合 vq_lookup 的逻辑)
//        for(int t=0; t<Ti; t++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_out = 0;
//            for(int k=0; k<64; k++) {
//                float f_res = (float)cb_local[best_idx[t]][k] * dec_scale_inv;
//                int8_t out_val = (f_res > 127.0f) ? 127 : (f_res < -128.0f) ? -128 : (int8_t)f_res;
//                raw_out.range(8*k+7, 8*k) = out_val;
//            }
//            out_z_q[n + t] = raw_out;
//        }
//    }
//}
//}

//尽量用原版2604192207  60ms
//#include <ap_fixed.h>
//#include <ap_int.h>
//
//// 1. 定义与原始 IP 完全一致的类型和宏
//typedef ap_fixed<16, 7, AP_RND, AP_SAT> vq_data_t; // 来自 vq.h
//typedef ap_int<16> lookup_data_t;                  // 来自 lookup.h
//
//// 原始分块参数
//#define TP 16
//#define TN 1
//#define TM 16
//#define TI 4
//#define INDEX_SIZE 21875 // 125 * 175
//
//// 2. 声明原始核心计算函数 (不带 M_AXI 接口)
//// 模仿 vq.cpp 中的 compute
//void compute_core(vq_data_t fm_in[TI][TP], vq_data_t fm_out[TI][TM][TP], vq_data_t cb[TM][TN]) {
//    #pragma HLS INLINE
//    for(int i=0; i<TP; i++){
//        for(int t=0; t<TI; t+=2) {
//            #pragma HLS PIPELINE II=1
//            for(int mm=0; mm<TM; mm++){
//                #pragma HLS UNROLL
//                vq_data_t diff1 = fm_in[t][i] - cb[mm][0];
//                fm_out[t][mm][i] += diff1 * diff1;
//                vq_data_t diff2 = fm_in[t+1][i] - cb[mm][0];
//                fm_out[t+1][mm][i] += diff2 * diff2;
//            }
//        }
//    }
//}
//
//extern "C" {
//void vq_accel(
//    vq_data_t* in_z,         // 原始定点输入
//    vq_data_t* codebook,     // 原始定点 Codebook
//    lookup_data_t* out_z_q   // 原始定点输出
//) {
//    // 原始接口定义
//    #pragma HLS INTERFACE m_axi port=in_z         offset=slave bundle=gmem0 depth=1400000 // 21875 * 64
//    #pragma HLS INTERFACE m_axi port=codebook     offset=slave bundle=gmem1 depth=32768
//    #pragma HLS INTERFACE m_axi port=out_z_q      offset=slave bundle=gmem2 depth=1400000
//
//    // 显式指定 return 和所有指针参数都在同一个 bundle 'control' 中
//    #pragma HLS INTERFACE s_axilite port=in_z         bundle=control
//    #pragma HLS INTERFACE s_axilite port=codebook     bundle=control
//    #pragma HLS INTERFACE s_axilite port=out_z_q      bundle=control
//    #pragma HLS INTERFACE s_axilite port=return       bundle=control
//
//    // 内部 Buffer
//    static vq_data_t cb_local[512][64];
//    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=BRAM
//
//    // 桥接用的索引 Buffer
//    unsigned short index_tmp[INDEX_SIZE];
//    #pragma HLS BIND_STORAGE variable=index_tmp type=RAM_2P impl=BRAM
//
//    // --- 第一部分：VQ 寻优逻辑 (模仿 vq.cpp) ---
//    // 预加载 Codebook
//    for(int c=0; c<512; c++) {
//        for(int k=0; k<64; k++) {
//            #pragma HLS PIPELINE II=1
//            cb_local[c][k] = codebook[c * 64 + k];
//        }
//    }
//
//    // 寻找索引
//    search_loop: for(int n=0; n<INDEX_SIZE; n+=TI) {
//        vq_data_t fm_in_buff[TI][64];
//        #pragma HLS ARRAY_PARTITION variable=fm_in_buff complete dim=1
//
//        // 加载输入
//        for(int t=0; t<TI; t++) {
//            for(int k=0; k<64; k++) {
//                #pragma HLS PIPELINE II=1
//                fm_in_buff[t][k] = in_z[(n + t) * 64 + k];
//            }
//        }
//
//        vq_data_t dist_min[TI] = {127.0, 127.0, 127.0, 127.0};
//        unsigned short best_idx[TI] = {0,0,0,0};
//
//        // 核心寻优
//        for(int c=0; c<512; c++) {
//            #pragma HLS PIPELINE II=1
//            vq_data_t acc[TI] = {0,0,0,0};
//            for(int k=0; k<64; k++) {
//                #pragma HLS UNROLL
//                for(int t=0; t<TI; t++) {
//                    vq_data_t diff = fm_in_buff[t][k] - cb_local[c][k];
//                    acc[t] += diff * diff;
//                }
//            }
//            for(int t=0; t<TI; t++) {
//                if(acc[t] < dist_min[t]) {
//                    dist_min[t] = acc[t];
//                    best_idx[t] = (unsigned short)c;
//                }
//            }
//        }
//
//        // 存储索引到中间 Buffer
//        for(int t=0; t<TI; t++) {
//            index_tmp[n + t] = best_idx[t];
//        }
//    }
//
//    // --- 第二部分：Lookup 查表逻辑 (模仿 lookup.cpp) ---
//    // 直接根据 index_tmp 从 cb_local 查表并输出
//    lookup_loop: for(int n=0; n<INDEX_SIZE; n++) {
//        unsigned short idx = index_tmp[n];
//        for(int k=0; k<64; k++) {
//            #pragma HLS PIPELINE II=1
//            // 直接透传定点数据，不涉及任何 float 转换
//            out_z_q[n * 64 + k] = (lookup_data_t)cb_local[idx][k].to_int();
//        }
//    }
//}
//}

//202604220926  30ms
//#include "vq_accel.h"
//#include <ap_fixed.h>
//
//// 1. 外部接口与最终输出维持带饱和舍入的高精度类型
//typedef ap_fixed<16, 7, AP_RND, AP_SAT> data_t;
//
//// 2. [优化] 内部计算用定点类型，去除 AP_RND 和 AP_SAT 以节省海量 LUT
//typedef ap_fixed<16, 7> calc_t;
//
//// 3. [优化] 累加器采用更宽的类型，自然避免溢出，代替 AP_SAT 逻辑
//typedef ap_fixed<32, 16> acc_t;
//
//#define Tp 16
//#define Tn 1
//#define Tm 16
//#define Ti 4
//#define MAX_LEN 1024
//
//// --- 声明子函数 (保持不变，或也可以同步替换 data_t 为 calc_t 以省资源) ---
//void compute_core(calc_t fm_in_buff[Ti][Tp], calc_t fm_out_buff[Ti][Tm][Tp], calc_t cb_buff[Tm][Tn]) {
//    #pragma HLS INLINE
//    for(int i=0; i<Tp; i++){
//        for(int t=0; t<Ti; t+=2) {
//            #pragma HLS PIPELINE II=1
//            for(int mm=0; mm<Tm; mm++){
//                #pragma HLS UNROLL
//                calc_t diff1 = fm_in_buff[t][i] - cb_buff[mm][0];
//                fm_out_buff[t][mm][i] += diff1 * diff1;
//                calc_t diff2 = fm_in_buff[t+1][i] - cb_buff[mm][0];
//                fm_out_buff[t+1][mm][i] += diff2 * diff2;
//            }
//        }
//    }
//}
//
//extern "C" {
//void vq_accel(
//    bus_t *in_z,
//    bus_t *in_codebook,
//    bus_t *out_z_q,
//    float enc_scale,
//    float dec_scale_inv
//) {
//    #pragma HLS INTERFACE m_axi port=in_z         offset=slave bundle=gmem0 depth=21875
//    #pragma HLS INTERFACE m_axi port=in_codebook  offset=slave bundle=gmem1 depth=2048
//    #pragma HLS INTERFACE m_axi port=out_z_q      offset=slave bundle=gmem2 depth=21875
//    #pragma HLS INTERFACE s_axilite port=in_z bundle=control
//    #pragma HLS INTERFACE s_axilite port=in_codebook bundle=control
//    #pragma HLS INTERFACE s_axilite port=out_z_q bundle=control
//    #pragma HLS INTERFACE s_axilite port=enc_scale bundle=control
//    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
//    #pragma HLS INTERFACE s_axilite port=return bundle=control
//
//    // [优化] 将类型改为 calc_t
//    static calc_t cb_local[512][64];
//    // [优化] 强制使用 BRAM（如果想要使用 URAM，可以将 impl=BRAM 改为 impl=URAM）
//    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_1P impl=BRAM
//    // [非常重要] 显式将维度 2 完全展开，确保 HLS 会生成 64 块 BRAM 进行并行读取，而不是将数组打散为 FF 寄存器！
//    #pragma HLS ARRAY_PARTITION variable=cb_local complete dim=2
//
//    // 1. 加载 Codebook
//    for(int i=0; i<512; i++) {
//        for(int j=0; j<4; j++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw = in_codebook[i*4 + j];
//            for(int v=0; v<16; v++) {
//                uint32_t bits = raw.range(32*v+31, 32*v);
//                float f_val = *(float*)(&bits);
//                cb_local[i][j*16 + v] = (calc_t)f_val;
//            }
//        }
//    }
//
//    main_loop: for(int n=0; n<21875; n+=Ti) {
//
//        calc_t fm_in_buff[Ti][64];
//        #pragma HLS ARRAY_PARTITION variable=fm_in_buff complete dim=1
//        // [优化] 同样确保输入缓冲的并行读取采用寄存器或小 BRAM 阵列规范化
//        #pragma HLS ARRAY_PARTITION variable=fm_in_buff complete dim=2
//
//        acc_t dist_min[Ti];   // [优化] 改用大位宽累加类型
//        unsigned short best_idx[Ti];
//        #pragma HLS ARRAY_PARTITION variable=dist_min complete
//        #pragma HLS ARRAY_PARTITION variable=best_idx complete
//
//        // 加载输入
//        for(int t=0; t<Ti; t++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw = in_z[n + t];
//            for(int k=0; k<64; k++) {
//                int8_t val = (int8_t)raw.range(8*k+7, 8*k);
//                fm_in_buff[t][k] = (calc_t)((float)val * enc_scale);
//            }
//        }
//
//        // 初始化最小值
//        for(int t=0; t<Ti; t++) {
//            dist_min[t] = 1000000.0; // [注意] 由于没有饱和机制，必须赋一个足够大的初始值
//            best_idx[t] = 0;
//        }
//
//        // 3. 核心 VQ 搜索
//        search_loop: for(int c=0; c<512; c++) {
//            #pragma HLS PIPELINE II=1
//            acc_t acc[Ti] = {0,0,0,0};
//            #pragma HLS ARRAY_PARTITION variable=acc complete
//
//            for(int k=0; k<64; k++) {
//                #pragma HLS UNROLL
//                calc_t cb_val = cb_local[c][k];
//                for(int t=0; t<Ti; t++) {
//                    calc_t diff = fm_in_buff[t][k] - cb_val;
//
//                    // [优化] 强制要求差值的平方采用 DSP 计算 (而不是 LUT 搭建的乘法器)
//                    acc_t diff_sq = (acc_t)diff * (acc_t)diff;
//                    #pragma HLS BIND_OP variable=diff_sq op=mul impl=dsp
//
//                    // [优化] 强制要求累加树使用 DSP 或 Fabric 进行平衡
//                    acc[t] += diff_sq;
//                }
//            }
//
//            for(int t=0; t<Ti; t++) {
//                if(acc[t] < dist_min[t]) {
//                    dist_min[t] = acc[t];
//                    best_idx[t] = c;
//                }
//            }
//        }
//
//        // 4. 写回转换
//        for(int t=0; t<Ti; t++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_out = 0;
//            for(int k=0; k<64; k++) {
//                float f_res = (float)cb_local[best_idx[t]][k] * dec_scale_inv;
//                int8_t out_val = (f_res > 127.0f) ? 127 : (f_res < -128.0f) ? -128 : (int8_t)f_res;
//                raw_out.range(8*k+7, 8*k) = out_val;
//            }
//            out_z_q[n + t] = raw_out;
//        }
//    }
//}
//}

//29ms 低面积  但是是fixed，结果正确
//#include "vq_accel.h"
//#include <ap_fixed.h>
//
//// 【核心类型】18位宽，完美贴合 DSP48 (27x18)
//typedef ap_fixed<18, 6> calc_t;
//typedef ap_fixed<42, 18> acc_t;
//
//extern "C" {
//void vq_accel(
//    bus_t *in_z,
//    bus_t *in_codebook,
//    bus_t *out_z_q,
//    float enc_scale,
//    float dec_scale_inv
//) {
//    #pragma HLS INTERFACE m_axi port=in_z         offset=slave bundle=gmem0 depth=21875
//    #pragma HLS INTERFACE m_axi port=in_codebook  offset=slave bundle=gmem1 depth=2048
//    #pragma HLS INTERFACE m_axi port=out_z_q      offset=slave bundle=gmem2 depth=21875
//    #pragma HLS INTERFACE s_axilite port=in_z bundle=control
//    #pragma HLS INTERFACE s_axilite port=in_codebook bundle=control
//    #pragma HLS INTERFACE s_axilite port=out_z_q bundle=control
//    #pragma HLS INTERFACE s_axilite port=enc_scale bundle=control
//    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
//    #pragma HLS INTERFACE s_axilite port=return bundle=control
//
//    // 转换缩放系数
//    calc_t fx_enc_scale = (calc_t)enc_scale;
//    // dec_scale_inv 为 32.0，用 16 位宽、8 位整数完美容纳
//    ap_fixed<16, 8> fx_dec_scale_inv = (ap_fixed<16, 8>)dec_scale_inv;
//
//    static calc_t cb_local[512][64];
//    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=URAM
//    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete
//
//    // 1. 预加载 Codebook
//    load_cb: for(int i=0; i<512; i++) {
//        for(int j=0; j<4; j++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_cb = in_codebook[i * 4 + j];
//            for(int v=0; v<16; v++) {
//                uint32_t bits = raw_cb.range(32*v+31, 32*v);
//                float f_val = *(float*)(&bits);
//                cb_local[i][j*16 + v] = (calc_t)f_val;
//            }
//        }
//    }
//
//    // 2. 极致主循环
//    main_loop: for(int n=0; n<21875; n+=TI) {
//
//        calc_t z_v[TI][64];
//        #pragma HLS ARRAY_PARTITION variable=z_v complete dim=0
//
//        read_z: for(int t=0; t<TI; t++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_in_bus = in_z[n + t];
//            for (int k = 0; k < 64; k++) {
//                int8_t tmp_val = (int8_t)raw_in_bus.range(8*k+7, 8*k);
//
//                // 【修复元凶！】先放入10位宽纯整数类型，彻底杜绝 127 变成 -1 的溢出截断！
//                ap_fixed<10, 10> safe_i8 = tmp_val;
//                z_v[t][k] = (calc_t)(safe_i8 * fx_enc_scale);
//            }
//        }
//
//        acc_t min_dist[TI];
//        int best_idx[TI];
//        #pragma HLS ARRAY_PARTITION variable=min_dist complete
//        #pragma HLS ARRAY_PARTITION variable=best_idx complete
//
//        init_min: for(int t=0; t<TI; t++) {
//            #pragma HLS UNROLL
//            min_dist[t] = 100000.0;
//            best_idx[t] = 0;
//        }
//
//        // 3. VQ 核心计算
//        find_min: for(int c=0; c<512; c++) {
//            #pragma HLS PIPELINE II=1 rewind
//
//            acc_t dist_accum[TI];
//            #pragma HLS ARRAY_PARTITION variable=dist_accum complete
//            for(int t=0; t<TI; t++) {
//                dist_accum[t] = 0;
//            }
//
//            calc_dist: for(int j=0; j<4; j++) {
//                for(int t=0; t<TI; t++) {
//                    acc_t partial_sum = 0;
//
//                    #pragma HLS EXPRESSION_BALANCE
//                    for(int v=0; v<16; v++) {
//                        int k = j*16 + v;
//                        calc_t diff = z_v[t][k] - cb_local[c][k];
//
//                        // DSP 完美乘法
//                        acc_t sq = diff * diff;
//                        #pragma HLS BIND_OP variable=sq op=mul impl=dsp
//
//                        partial_sum += sq;
//                    }
//                    dist_accum[t] += partial_sum;
//                }
//            }
//
//            update_min: for(int t=0; t<TI; t++) {
//                if(dist_accum[t] < min_dist[t]) {
//                    min_dist[t] = dist_accum[t];
//                    best_idx[t] = c;
//                }
//            }
//        }
//
//        // 4. 写回转换
//        write_out: for(int t=0; t<TI; t++) {
//            #pragma HLS PIPELINE II=1
//            bus_t raw_out_bus = 0;
//            convert_out: for (int k = 0; k < 64; k++) {
//                // 【绝杀 LUT】消灭最后 320 个浮点乘法！纯定点算完直接交给硬件进行截断饱和 (AP_SAT)
//                ap_fixed<24, 12> val_scaled = cb_local[best_idx[t]][k] * fx_dec_scale_inv;
//
//                ap_fixed<8, 8, AP_RND, AP_SAT> sat_val = val_scaled;
//                raw_out_bus.range(8*k+7, 8*k) = (ap_uint<8>)sat_val;
//            }
//            out_z_q[n + t] = raw_out_bus;
//        }
//    }
//}
//}

//36ms 增加load_cb_flag，避免重复读取codebook，改变gmem，结果正确
//#include "vq_accel.h"
//#include <string.h> // 必须包含 memcpy
//
//extern "C" {
//void vq_accel(
//    bus_t *in_z,
//    bus_t *in_codebook,
//    bus_t *out_z_q,
//    float enc_scale,
//    float dec_scale_inv,
//    int   load_cb_flag
//) {
//    // 接口定义（维持双 Bundle 架构）
//    #pragma HLS INTERFACE m_axi port=in_z         offset=slave bundle=gmem_in  depth=21875 num_read_outstanding=16
//    #pragma HLS INTERFACE m_axi port=in_codebook  offset=slave bundle=gmem_in  depth=2048 num_read_outstanding=16
//    #pragma HLS INTERFACE m_axi port=out_z_q      offset=slave bundle=gmem_out depth=21875 num_read_outstanding=16
//
//    #pragma HLS INTERFACE s_axilite port=in_z         bundle=control
//    #pragma HLS INTERFACE s_axilite port=in_codebook  bundle=control
//    #pragma HLS INTERFACE s_axilite port=out_z_q      bundle=control
//    #pragma HLS INTERFACE s_axilite port=enc_scale     bundle=control
//    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
//    #pragma HLS INTERFACE s_axilite port=load_cb_flag  bundle=control
//    #pragma HLS INTERFACE s_axilite port=return        bundle=control
//
//    // 存储码本的静态 URAM
//    static calc_t cb_local[512][64];
//    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=URAM
//    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete
//
//    // --- 修复点 1: 显式码本搬运逻辑 ---
//    if (load_cb_flag == 1) {
//        // 使用 memcpy 强制触发 AXI Burst 读取，确保搬运成功
//        bus_t temp_cb_buf[2048];
//        memcpy(temp_cb_buf, (const bus_t*)in_codebook, 2048 * sizeof(bus_t));
//
//        load_cb_inner: for(int i=0; i<512; i++) {
//            #pragma HLS PIPELINE II=1
//            for(int j=0; j<4; j++) {
//                bus_t raw_val = temp_cb_buf[i * 4 + j];
//                for(int v=0; v<16; v++) {
//                    uint32_t bits = raw_val.range(32*v+31, 32*v);
//                    cb_local[i][j*16 + v] = (calc_t)(*(float*)(&bits));
//                }
//            }
//        }
//    }
//
//    // --- 修复点 2: 使用显式 Burst 读取输入数据 ---
//    main_loop: for(int n=0; n<NUM_Z; n+=TI) {
//        bus_t in_burst[TI];
//        #pragma HLS ARRAY_PARTITION variable=in_burst complete
//        memcpy(in_burst, (const bus_t*)(in_z + n), TI * sizeof(bus_t));
//
//        calc_t z_v[TI][64];
//        #pragma HLS ARRAY_PARTITION variable=z_v complete dim=0
//
//        for(int t=0; t<TI; t++) {
//            #pragma HLS UNROLL
//            for (int k = 0; k < 64; k++) {
//                int8_t tmp = (int8_t)in_burst[t].range(8*k+7, 8*k);
//                z_v[t][k] = (calc_t)((ap_fixed<10, 10>)tmp * (calc_t)enc_scale);
//            }
//        }
//
//        acc_t min_dist[TI];
//        int best_idx[TI];
//        #pragma HLS ARRAY_PARTITION variable=min_dist complete
//        #pragma HLS ARRAY_PARTITION variable=best_idx complete
//
//        find_min: for(int c=0; c<512; c++) {
//            #pragma HLS PIPELINE II=1 rewind
//            for(int t=0; t<TI; t++) {
//                acc_t dist_accum = 0;
//                for(int k=0; k<64; k++) {
//                    calc_t diff = z_v[t][k] - cb_local[c][k];
//                    dist_accum += diff * diff;
//                }
//                // 修复点 3: 显式初始化第一个比较值
//                if(c == 0 || dist_accum < min_dist[t]) {
//                    min_dist[t] = dist_accum;
//                    best_idx[t] = c;
//                }
//            }
//        }
//
//        // --- 修复点 4: 显式 Burst 写回数据 ---
//        bus_t out_burst[TI];
//        #pragma HLS ARRAY_PARTITION variable=out_burst complete
//        for(int t=0; t<TI; t++) {
//            #pragma HLS UNROLL
//            bus_t raw_out = 0;
//            for (int k = 0; k < 64; k++) {
//                ap_fixed<24, 12> val = cb_local[best_idx[t]][k] * (ap_fixed<16,8>)dec_scale_inv;
//                ap_fixed<8, 8, AP_RND, AP_SAT> sat = val;
//                raw_out.range(8*k+7, 8*k) = (ap_uint<8>)sat;
//            }
//            out_burst[t] = raw_out;
//        }
//        memcpy((bus_t*)(out_z_q + n), out_burst, TI * sizeof(bus_t));
//    }
//}
//}

#include "vq_accel.h"
#include <ap_fixed.h>
#include <stdint.h>

extern "C" {
void vq_accel(
    bus_t *in_z,
    bus_t *in_codebook,
    bus_t *out_z_q,
    float enc_scale,
    float dec_scale_inv
) {
    #pragma HLS INTERFACE m_axi port=in_z         offset=slave bundle=gmem0 depth=24576
    #pragma HLS INTERFACE m_axi port=in_codebook  offset=slave bundle=gmem1 depth=2048
    #pragma HLS INTERFACE m_axi port=out_z_q      offset=slave bundle=gmem2 depth=24576
    #pragma HLS INTERFACE s_axilite port=in_z bundle=control
    #pragma HLS INTERFACE s_axilite port=in_codebook bundle=control
    #pragma HLS INTERFACE s_axilite port=out_z_q bundle=control
    #pragma HLS INTERFACE s_axilite port=enc_scale bundle=control
    #pragma HLS INTERFACE s_axilite port=dec_scale_inv bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // ----------------------------------------
    // 量化参数
    // ----------------------------------------
    calc_t fx_enc_scale = (calc_t)enc_scale;
    ap_fixed<16, 8> fx_dec_scale_inv = (ap_fixed<16, 8>)dec_scale_inv;

    // ----------------------------------------
    // 本地 codebook
    // ----------------------------------------
    static calc_t cb_local[NUM_C][DIM];
    #pragma HLS BIND_STORAGE variable=cb_local type=RAM_2P impl=URAM
    #pragma HLS ARRAY_PARTITION variable=cb_local dim=2 complete

    // ----------------------------------------
    // 1) 加载 Codebook
    // 每个 codeword 64 float = 4 * 512bit
    // ----------------------------------------
    load_cb:
    for (int i = 0; i < NUM_C; i++) {
        for (int j = 0; j < 4; j++) {
            #pragma HLS PIPELINE II=1
            bus_t raw_cb = in_codebook[i * 4 + j];

            for (int v = 0; v < 16; v++) {
                uint32_t bits = (uint32_t)raw_cb.range(32 * v + 31, 32 * v);

                // 保持与你 700x500 正确版一致
                float f_val = *(float*)(&bits);
                cb_local[i][j * 16 + v] = (calc_t)f_val;
            }
        }
    }

    // ----------------------------------------
    // 2) 主循环
    // 每个 in_z[n] / out_z_q[n] 对应一个 64维 int8 向量
    // ----------------------------------------
    main_loop:
    for (int n = 0; n < NUM_Z; n += TI) {

        calc_t z_v[TI][DIM];
        #pragma HLS ARRAY_PARTITION variable=z_v complete dim=0

        // ----------------------------------------
        // 2.1 读取输入 latent，并反量化到 calc_t
        // ----------------------------------------
        read_z:
        for (int t = 0; t < TI; t++) {
            #pragma HLS PIPELINE II=1
            bus_t raw_in_bus = in_z[n + t];

            for (int k = 0; k < DIM; k++) {
                int8_t tmp_val = (int8_t)raw_in_bus.range(8 * k + 7, 8 * k);

                // 保持 700x500 正确版的安全转换
                ap_fixed<10, 10> safe_i8 = tmp_val;
                z_v[t][k] = (calc_t)(safe_i8 * fx_enc_scale);
            }
        }

        // ----------------------------------------
        // 2.2 初始化最小距离
        // ----------------------------------------
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

        // ----------------------------------------
        // 2.3 遍历 codebook，找最近邻
        // ----------------------------------------
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

        // ----------------------------------------
        // 2.4 写回：codebook float -> decoder int8 输入
        // ----------------------------------------
        write_out:
        for (int t = 0; t < TI; t++) {
            #pragma HLS PIPELINE II=1
            bus_t raw_out_bus = 0;

            convert_out:
            for (int k = 0; k < DIM; k++) {
                ap_fixed<24, 12> val_scaled = cb_local[best_idx[t]][k] * fx_dec_scale_inv;
                ap_fixed<8, 8, AP_RND, AP_SAT> sat_val = val_scaled;
                raw_out_bus.range(8 * k + 7, 8 * k) = (ap_uint<8>)sat_val;
            }

            out_z_q[n + t] = raw_out_bus;
        }
    }
}
}
