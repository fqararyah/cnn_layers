#ifndef PW_CONV_H
#define PW_CONV_H

#include "../../basic_defs/basic_defs_glue.h"

void pw_conv(weights_grp_dt *weights, fms_dt channels[max_fms_size],
			 fms_dt result[max_fms_size], int layer, const int layer_conv_d,
			 const int layer_num_fils, const int num_of_tiles_d_in,
			 const int num_of_tiles_d_out, const int num_of_tiles_h,
			 const int num_of_tiles_w, fms_dt tmp_channels[max_tmp_fms_size],
			 int read_write, const int num_of_weight_groups, const int direction, const int layer_weights_offset, const int layer_relu,
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
			 int layer, const int layer_conv_d,
			 const int layer_num_fils, const int num_of_tiles_d_in,
			 const int num_of_tiles_d_out, const int num_of_tiles_h,
			 const int num_of_tiles_w, fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
			 int read_write, const int num_of_weight_groups, const int direction, const int layer_weights_offset, const int layer_relu,
			 const fused_scales_dt fused_scales[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
			 const relu_6_fused_scales_dt relu_6_fused_scales[],
			 const biases_dt fused_zero_points[],
			 const fused_scales_dt fused_scales_part2[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
			 const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
			 const biases_dt fused_zero_points_part2[]);

void pw_conv_eng(fms_dt channels_tile[pw_tile_d][pw_tile_h][pw_tile_w],
				 weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
				 pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
				 int starting_conv_d, int starting_filter, const int layer_conv_d,
				 const int layer_num_fils);
#endif
