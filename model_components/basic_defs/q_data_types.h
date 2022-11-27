#ifndef Q_DATA_TYPES
#define Q_DATA_TYPES

#include "ap_int.h"
#include "ap_fixed.h"

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
const int weights_group_items = 512 / weights_dt_width;//TODO double check that 512 is the max that can be loaded at once

//fms
const int fms_dt_width = 8;
const int fms_dt_offset = fms_dt_width - 1;
const int input_image_group_items = 512 / fms_dt_width;

//scales, zero points, and biases
const int scales_bit_width = 48;//48
const int scales_integer_part_width = 1;
const int fused_scales_bit_width = 36;//48
const int fused_scales_integer_part_width = 1;
const int rec_scales_bit_width = 24;//48
const int rec_scales_integer_part_width = 8;

const int biases_bit_width = 32;

//pss
const int pss_dt_width = 32;//weights_dt_width + fms_dt_width + 10; // 10 is log(1024, 2) since 1024 is the depth of the deepest filter
const int pss_dt_offset = pss_dt_width - 1;
const int first_conv_pss_width = fms_dt_width + layer_0_weights_dt_width + 4;
const int dw_pss_dt_width = 32;//dw_weights_dt_width + fms_dt_width + 4; // 4 is ceil(log(9, 2))
const int dw_pss_dt_offset = dw_pss_dt_width - 1;					// 11;
const int fc_weights_dt_width = 8;
const int fc_out_dt_width = fms_dt_width + fc_weights_dt_width + 11; //11 is ceil(log(fc_layer_input_size))

typedef ap_int<layer_0_weights_dt_width> layer_0_weights_dt;
typedef ap_int<weights_dt_width> weights_dt;
typedef ap_int<dw_weights_dt_width> dw_weights_dt;
typedef ap_int<fms_dt_width> fms_dt;
typedef ap_int<pss_dt_width> pss_dt;	   // partial sums
typedef ap_fixed<pss_dt_width + 10, pss_dt_width> pss_f_dt; //+ 16
typedef ap_int<dw_pss_dt_width> dw_pss_dt; // partial sums
typedef ap_fixed<dw_pss_dt_width + 10, dw_pss_dt_width> dw_pss_f_dt; // + 16
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
//typedef scales_dt rec_scales_dt;
typedef ap_ufixed<rec_scales_bit_width, rec_scales_integer_part_width> rec_scales_dt;

typedef int biases_dt;
struct fms_quantization_scheme {
	 fms_dt ofm_zero_point;
	 scales_dt ifm_scale;
	 rec_scales_dt ofm_scale_rec;
	 biases_dt fused_zero_point;
	 fused_scales_dt fused_scales;
	//const biases_dt bias;
};

const pss_f_dt quant_half = (pss_f_dt) 0.5;
const dw_pss_f_dt quant_dw_half = (dw_pss_f_dt) 0.5;
const fms_dt QUANTIZATION_MAX = 127;
const fms_dt QUANTIZATION_MIN = -128;

#endif
