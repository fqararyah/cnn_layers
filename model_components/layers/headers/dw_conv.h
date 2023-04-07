#ifndef DW_CONV_H
#define DW_CONV_H

#include "../../basic_defs/basic_defs_glue.h"

void dw_conv_3x3(const dw_weights_dt weights[][3 * 3],
				 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
				 const int layer, const int layer_conv_d, const int layer_ifm_width, const int layer_ifm_height,
				 const int num_of_tiles_d,
				 const int num_of_ifms_tiles_h, const int num_of_ifms_tiles_w,
				 const int num_of_tiles_h, const int num_of_tiles_w,
				 const int strides, const int padding_left, const int padding_right, const int padding_top,
				 const fused_scales_dt fused_scales[],
				 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
				 const relu_6_fused_scales_dt relu_6_fused_scales[], const biases_dt fused_zero_points[],
				 const fused_scales_dt fused_scales_part2[],
				 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
				 const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
				 const biases_dt fused_zero_points_part2[]);

//V2
void dw_conv_3x3(const dw_weights_dt weights[][3 * 3],
					fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
					fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
					const int layer,
					const layer_specs layer_specs_struct,
					const fused_scales_dt fused_scales[],
					const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
					const relu_6_fused_scales_dt relu_6_fused_scales[], const biases_dt fused_zero_points[],
					const fused_scales_dt fused_scales_part2[],
					const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
					const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
					const biases_dt fused_zero_points_part2[]);

void dw_conv_5x5(dw_weights_dt weights[max_conv_d][5][5],
				 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
				 const int layer, const int layer_conv_d, const int num_of_tiles_d,
				 const int num_of_tiles_h, const int num_of_tiles_w, const int strides);

void dw_conv_7x7(dw_weights_dt weights[max_conv_d][7][7],
				 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
				 const int layer, const int layer_conv_d, const int num_of_tiles_d,
				 const int num_of_tiles_h, const int num_of_tiles_w, const int strides);

#endif
