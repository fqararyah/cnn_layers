
#include "../headers/layers_imp_common_includes.h"
#include "../headers/pw_conv.h"

void pw_conv(weights_grp_dt *weights,
			 fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
			 fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
			 fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
			 int layer, int read_write, const int direction,
			 const layer_specs layer_specs_struct,
			 const fused_scales_dt fused_scales[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
			 const relu_6_fused_scales_dt relu_6_fused_scales[],
			 const biases_dt fused_zero_points[],
			 const fused_scales_dt fused_scales_part2[],
			 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
			 const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
			 const biases_dt fused_zero_points_part2[])
{
// #pragma HLS INLINE off

// 	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d];
// #pragma HLS ARRAY_PARTITION variable = weights_tile complete dim = 1
// #pragma HLS ARRAY_PARTITION variable = weights_tile cyclic dim = 2 factor = num_of_weights_in_the_same_filter_and_group

// #if HW == FPGA
// 	weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile];
// 	fill_layer_weight_groups_tile_off_chip(weights, weight_groups_buffer, 0,
// 										   layer_conv_d, num_of_weight_groups, layer_weights_offset,
// 										   layer_num_fils);
// #elif HW == CPU
// 	fill_layers_weights_cpu(weights,
// 							weights_tile,
// 							0 * pw_conv_parallelism_out, layer_conv_d,
// 							layer_weights_offset, layer_num_fils);
// #endif

// 	biases_dt fused_zero_points_buffer[pw_conv_parallelism_out];
// 	fused_scales_dt fused_scales_buffer[pw_conv_parallelism_out];
// 	fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[pw_conv_parallelism_out];
// 	relu_6_fused_scales_dt relu_6_fused_scales_buffer[pw_conv_parallelism_out];

// 	const int current_layer_fused_parameters_offset = layers_fused_parameters_offsets[layer];

// conv2_ots_loop:
// 	for (int td_o = 0; td_o < num_of_tiles_d_out; td_o++)
// 	{
// 		if (current_layer_fused_parameters_offset < first_quantization_arrays_num_elements)
// 		{
// 			fill_fused_zero_points_buffer(fused_zero_points,
// 										  fused_zero_points_buffer, td_o * pw_conv_parallelism_out,
// 										  layer, current_layer_fused_parameters_offset);
// 			fill_fused_scales_buffer(fused_scales, fused_scales_buffer,
// 									 fused_scales_log_2_shifts, fused_scales_log_2_shifts_buffer,
// 									 relu_6_fused_scales, relu_6_fused_scales_buffer,
// 									 td_o * pw_conv_parallelism_out, layer, current_layer_fused_parameters_offset);
// 		}
// 		else
// 		{
// 			fill_fused_zero_points_buffer(fused_zero_points_part2,
// 										  fused_zero_points_buffer, td_o * pw_conv_parallelism_out,
// 										  layer, current_layer_fused_parameters_offset - first_quantization_arrays_num_elements);
// 			fill_fused_scales_buffer(fused_scales_part2, fused_scales_buffer,
// 									 fused_scales_log_2_shifts_part2, fused_scales_log_2_shifts_buffer,
// 									 relu_6_fused_scales_part2, relu_6_fused_scales_buffer,
// 									 td_o * pw_conv_parallelism_out, layer,
// 									 current_layer_fused_parameters_offset - first_quantization_arrays_num_elements);
// 		}

// #if HW == FPGA
// 		fill_weights_tile_from_weight_groups_tile(weight_groups_buffer,
// 												  weights_tile, td_o * pw_conv_parallelism_out, layer_conv_d,
// 												  num_of_weight_groups, layer_weights_offset);
// #endif
// 		do_conv(weights_tile, channels, result, layer, layer_conv_d,
// 				layer_num_fils, num_of_tiles_d_in, num_of_tiles_d_out,
// 				num_of_tiles_h, num_of_tiles_w, tmp_channels, read_write,
// 				num_of_weight_groups, direction, layer_weights_offset,
// 				layer_relu, fused_scales_buffer,
// 				fused_scales_log_2_shifts_buffer, relu_6_fused_scales_buffer,
// 				fused_zero_points_buffer, td_o);
// #if HW == FPGA
// 		fill_layer_weight_groups_tile_off_chip(weights, weight_groups_buffer,
// 											   (td_o + 1) * pw_conv_parallelism_out, layer_conv_d,
// 											   num_of_weight_groups, layer_weights_offset, layer_num_fils);
// #elif HW == CPU
// 		fill_layers_weights_cpu(weights,
// 								weights_tile,
// 								(td_o + 1) * pw_conv_parallelism_out, layer_conv_d,
// 								layer_weights_offset, layer_num_fils);
// #endif
// 	}
}
