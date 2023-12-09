#ifndef Q_DATA_TYPES
#define Q_DATA_TYPES

#if HW == _FPGA
#include "ap_int.h"
#include "ap_fixed.h"
#elif CPU
#include <cstdio>
#include <cinttypes>

using namespace std;

#endif

//input_image
const int input_image_dt_width = 8;
const int input_image_dt_offset = input_image_dt_width - 1;
const int input_image_pss_dt_width = input_image_dt_width + 8 + 5; // 5 is ceil(log(27, 2))

//weights
const int weights_dt_width = 8;
const int weights_dt_offset = weights_dt_width - 1;
const int layer_0_weights_dt_width = 8;
const int layer_0_weights_dt_offset = layer_0_weights_dt_width - 1;
const int dw_weights_dt_width = 8;
const int dw_weights_dt_offset = dw_weights_dt_width - 1;
#if HW == _FPGA
const int weights_group_items = 512 / weights_dt_width;//TODO double check that 512 is the max that can be loaded at once
#elif HW == CPU
const int weights_group_items = 1;
#endif
//fms
const int fms_dt_width = 8;
const int fms_dt_offset = fms_dt_width - 1;
#if HW == _FPGA
const int input_image_group_items = 512 / fms_dt_width;
#elif HW == CPU
const int input_image_group_items = 1;
#endif
//scales, zero points, and biases
const int scales_bit_width = 24;//48
const int scales_integer_part_width = 0;
const int fused_scales_bit_width = 16;//48
const int fused_scales_integer_part_width = 0;
const int relu_6_fused_scales_bit_width = 8;
const int layer_0_relu_6_fused_scales_bit_width = 8;
const int fused_scales_log_2_shifts_bit_width = 6;//2^6=64
const int rec_scales_bit_width = 24;//48
const int rec_scales_integer_part_width = 10;

const int biases_bit_width = 32;

//pss
const int pss_dt_width = 32;//weights_dt_width + fms_dt_width + 10; // 10 is log(1024, 2) since 1024 is the depth of the deepest filter
const int norm_act_pss_dt_width = relu_6_fused_scales_bit_width + 2;//+ 2 not + 1 because relu_6_fused_scales_bit_width is uint not int
const int layer_0_norm_act_pss_dt_width = layer_0_relu_6_fused_scales_bit_width + 2;//+2 look the comment in previous line
const int pss_dt_offset = pss_dt_width - 1;
const int first_conv_pss_width = pss_dt_width;
const int dw_pss_dt_width = 32;//dw_weights_dt_width + fms_dt_width + 4; // 4 is ceil(log(9, 2))
const int dw_pss_dt_offset = dw_pss_dt_width - 1;					// 11;
const int fc_weights_dt_width = 8;
const int fc_out_dt_width = fms_dt_width + fc_weights_dt_width + 11; //11 is ceil(log(fc_layer_input_size))

#if HW == _FPGA
typedef ap_int<layer_0_weights_dt_width> layer_0_weights_dt;
typedef ap_int<weights_dt_width> weights_dt;
typedef ap_int<dw_weights_dt_width> dw_weights_dt;
typedef ap_int<fms_dt_width> fms_dt;
typedef ap_int<pss_dt_width> pss_dt;	   // partial sums
typedef ap_int<norm_act_pss_dt_width> norm_act_pss_dt;
typedef ap_int<layer_0_norm_act_pss_dt_width> layer_0_norm_act_pss_dt;
typedef ap_fixed<pss_dt_width + 10, pss_dt_width> pss_f_dt; //+ 16
typedef ap_int<dw_pss_dt_width> dw_pss_dt; // partial sums
typedef ap_fixed<dw_pss_dt_width + 16, dw_pss_dt_width> dw_pss_f_dt; // + 16
typedef ap_int<first_conv_pss_width> first_conv_pss_dt;
typedef ap_uint<weights_group_items * weights_dt_width> weights_grp_dt;
typedef ap_uint<input_image_group_items * fms_dt_width> fms_grp_dt;
typedef ap_uint<11> counters_dt;
typedef ap_uint<input_image_dt_width> input_image_dt;
typedef ap_int<input_image_pss_dt_width> input_image_pss_dt;
typedef ap_int<fc_weights_dt_width> fc_weights_dt;
typedef ap_int<fc_out_dt_width> fc_out_dt;

typedef ap_ufixed<scales_bit_width, scales_integer_part_width> scales_dt;
//typedef scales_dt fused_scales_dt;
typedef ap_ufixed<fused_scales_bit_width, fused_scales_integer_part_width> fused_scales_dt;
typedef ap_ufixed<24, 4>  pooling_fused_scales_dt;
typedef ap_uint<fused_scales_log_2_shifts_bit_width> fused_scales_log_2_shifts_dt;
typedef ap_uint<relu_6_fused_scales_bit_width> relu_6_fused_scales_dt;
typedef ap_uint<layer_0_relu_6_fused_scales_bit_width> layer_0_relu_6_fused_scales_dt;
//typedef scales_dt rec_scales_dt;
typedef ap_ufixed<rec_scales_bit_width, rec_scales_integer_part_width> rec_scales_dt;
#elif HW == CPU
typedef int8_t layer_0_weights_dt;
typedef int8_t weights_dt;
typedef weights_dt weights_grp_dt;
typedef int8_t dw_weights_dt;
typedef int8_t fms_dt;
typedef fms_dt fms_grp_dt;
typedef int64_t pss_dt;	   // partial sums
typedef int64_t norm_act_pss_dt;
typedef int64_t layer_0_norm_act_pss_dt;
typedef double pss_f_dt; //+ 16
typedef int64_t dw_pss_dt; // partial sums
typedef double dw_pss_f_dt; // + 16
typedef int64_t first_conv_pss_dt;
typedef uint8_t input_image_dt;
typedef int8_t fc_weights_dt;
typedef int8_t fc_out_dt;

typedef float scales_dt;
//typedef scales_dt fused_scales_dt;
typedef float fused_scales_dt;
typedef float  pooling_fused_scales_dt;
typedef uint8_t fused_scales_log_2_shifts_dt;
typedef uint8_t relu_6_fused_scales_dt;
typedef uint8_t layer_0_relu_6_fused_scales_dt;
//typedef scales_dt rec_scales_dt;
typedef float rec_scales_dt;
#endif

typedef int biases_dt;

struct fms_quantization_scheme {
	 fms_dt ofm_zero_point;
	 scales_dt ifm_scale;
	 rec_scales_dt ofm_scale_rec;
	 scales_dt ofm_scale;
	 biases_dt fused_zero_point;
	 fused_scales_dt fused_scales;
	 fused_scales_log_2_shifts_dt fused_scales_log_2_shift;
	 relu_6_fused_scales_dt relu_6_fused_scale;
	 layer_0_relu_6_fused_scales_dt layer_0_relu_6_fused_scale;
	//const biases_dt bias;
};

const pss_f_dt quant_half = 0.5;
const dw_pss_f_dt quant_dw_half = 0.5;
const fms_dt QUANTIZATION_MAX = 127;
const fms_dt QUANTIZATION_MIN = -128;

#endif
