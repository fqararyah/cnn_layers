#ifndef _UTILS
#define _UTILS

#include <stdint.h>
#include <string>
#include "../basic_defs/basic_defs_glue.h"

#if MODEL_ID == MOB_V2
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#include "../model/headers/mob_v2_on_chip_weights.h"
#else
#include "../model/headers/mob_v2_on_chip_weights_v2.h"
#endif
#elif MODEL_ID == MOB_V2_0_5
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#include "../model/headers/mob_v2_0_5_on_chip_weights.h"
#else
#include "../model/headers/mob_v2_0_5_on_chip_weights_v2.h"
#endif
#elif MODEL_ID == MOB_V2_0_25
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#include "../model/headers/mob_v2_0_25_on_chip_weights.h"
#else
#include "../model/headers/mob_v2_0_25_on_chip_weights_v2.h"
#endif
#elif MODEL_ID == MOB_V2_0_75
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#include "../model/headers/mob_v2_0_75_on_chip_weights.h"
#else
#include "../model/headers/mob_v2_0_75_on_chip_weights_v2.h"
#endif
#endif

#include "../model/headers/model_glue.h"

using namespace std;

void fill_on_chip_weights_cpu(weights_grp_dt *on_chip_weights_src,
							  weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS]);

void fill_on_chip_weights_fpga(weights_grp_dt *on_chip_weights_src,
							   weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
							   const int starting_filter);

void fill_layers_weights_cpu(weights_dt *weights,
							 weights_dt weights_buffer[][max_conv_d],
							 int starting_filter, const int layer_depth,
							 const int layer_weights_offset,
							 const int layer_num_fils);

void fill_layers_weights_cpu_pw_conv(weights_dt *weights,
									 weights_dt weights_buffer[][max_conv_d][max_filter_area],
									 int starting_filter, const int layer_depth,
									 const int layer_weights_offset,
									 const int layer_num_fils);

void fill_layers_weights_cpu(weights_dt *weights,
							 weights_dt weights_buffer[][max_conv_d],
							 int starting_filter, const int layer_depth,
							 const int layer_weights_offset,
							 const int layer_num_fils);

void fill_layer_0_s_weights(
	layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][3][3]);

void fill_dw_layer_weights(
	const dw_weights_dt src[max_conv_d][max_conv_h * max_conv_w],
	dw_weights_dt dst[max_conv_d][max_conv_h * max_conv_w],
	const int conv_d, const int conv_h, const int conv_w);

void fill_weights_tile_off_chip(weights_grp_dt *weights,
								weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
								int starting_filter, const int layer_depth,
								const int num_of_weight_groups, const int layer_weights_offset);

void fill_layer_weight_groups_tile_off_chip(weights_grp_dt *weights,
											weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile],
											int starting_filter,
											const int layer_depth, const int num_of_weight_groups,
											const int layer_weights_offset,
											const int layer_num_fils);

void fill_weights_tile_from_weight_groups_tile(
	weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile],
	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
	const int layer_depth,
	const int num_of_weight_groups);

void fill_fused_zero_points_buffer(const biases_dt fused_zero_points[],
								   biases_dt fused_zero_points_buffer[pw_conv_parallelism_out],
								   int starting_d, int layer, const int current_layer_fused_parameters_offset);

void fill_fused_scales_buffer(const fused_scales_dt fused_scales[],
							  fused_scales_dt fused_scales_buffer[],int starting_d,
							  int layer, const int current_layer_fused_parameters_offset);

void fill_fused_scales_and_zero_points(
	const fused_scales_dt layer_fused_scales[],
	fused_scales_dt fused_scales[],
	const fused_scales_log_2_shifts_dt layer_fused_scales_log_2_shifts[],
	fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
	const biases_dt layer_fused_zero_points[],
	biases_dt fused_zero_points[], const int layer_num_filters);

void copy_channels_to_tmp_channels(fms_dt channels[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size]);
void copy_channels_to_tmp_channels(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
								   fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH]);

void fill_model_configs_list(const int model_configs_list_src[2 * max_conv_layers],
							 int model_configs_list[2 * max_conv_layers]);

#endif
