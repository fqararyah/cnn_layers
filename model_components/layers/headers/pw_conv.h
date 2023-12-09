#ifndef PW_CONV_H
#define PW_CONV_H

#include "../../basic_defs/basic_defs_glue.h"

void copy_pss_tile(
	pss_dt src_pss_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
	pss_dt dst_pss_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w]);

void pw_write_results_tile(
	fms_dt result_tile_scaled[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
	fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
	int tile_indx,
	fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
	pss_f_dt tmp_channels_scaled_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
	int starting_d,
	const int in_tile_h,
	const int in_tile_w,
	layer_specs layer_specs_struct);

void scale_pss_tile(fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
					pss_dt pss_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
					fms_dt result_tile_scaled[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
					layer_specs layer_specs_struct,
					fused_scales_dt fused_scales_buffer[pw_conv_parallelism_out],
					fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[pw_conv_parallelism_out],
					relu_6_fused_scales_dt relu_6_fused_scales_buffer[pw_conv_parallelism_out],
					biases_dt fused_zero_points_buffer[pw_conv_parallelism_out],
					const int tile_index);

void pw_conv(weights_grp_dt *weights, fms_dt channels[max_fms_size],
			 fms_dt result[max_fms_size],
			 fms_dt tmp_channels[max_tmp_fms_size],
			 int layer, const layer_specs layer_specs_struct,
			 const fused_scales_dt fused_scales[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
			 const relu_6_fused_scales_dt relu_6_fused_scales[],
			 const biases_dt fused_zero_points[],
			 const fused_scales_dt fused_scales_part2[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
			 const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
			 const biases_dt fused_zero_points_part2[]);

void pw_conv(weights_grp_dt *weights,
			 fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
			 fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
			 fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
			 int layer, const layer_specs layer_specs_struct,
			 const fused_scales_dt fused_scales[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
			 const relu_6_fused_scales_dt relu_6_fused_scales[],
			 const biases_dt fused_zero_points[],
			 const fused_scales_dt fused_scales_part2[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
			 const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
			 const biases_dt fused_zero_points_part2[],
			 const int model_configs_list[2 * max_conv_layers]);

void pw_conv_eng(fms_dt channels_tile[pw_tile_d][pw_tile_h][pw_tile_w],
				 weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
				 pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
				 int starting_conv_d, int starting_filter, const int layer_conv_d,
				 const int layer_num_fils);
#endif
