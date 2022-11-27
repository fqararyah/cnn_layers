#ifndef DW_CONV
#define DW_CONV

#include "../../basic_defs/basic_defs_glue.h"

void dw_conv_3x3(dw_weights_dt weights[max_conv_d][max_conv_h][max_conv_w],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int layer_width,const int layer_height,
		const int num_of_tiles_d, const int num_of_tiles_h,
		const int num_of_tiles_w, const int strides, const int padding_left, const int padding_top,
		const int direction, fused_scales_dt fused_scales[], biases_dt fused_zero_points[]);

void dw_conv_5x5(dw_weights_dt weights[max_conv_d][5][5],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int num_of_tiles_d,
		const int num_of_tiles_h, const int num_of_tiles_w, const int strides);

void dw_conv_7x7(dw_weights_dt weights[max_conv_d][7][7],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int num_of_tiles_d,
		const int num_of_tiles_h, const int num_of_tiles_w, const int strides);

#endif
