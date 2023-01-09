#ifndef BOTTLENECKS_CHAIN
#define BOTTLENECKS_CHAIN
#include "bottlenecks_parallelism.h"
#include "bottleneck_kernels.h"
#include "bottleneck.h"

const int chain_0_1_ifms_depth = bottleneck_0_ifms_depth;
const int chain_0_1_ifms_height = bottleneck_0_ifms_height;
const int chain_0_1_ifms_width = bottleneck_0_ifms_width;
const int chain_0_1_ofms_depth = bottleneck_1_ofms_depth;
const int chain_0_1_ofms_height = bottleneck_1_ofms_height;
const int chain_0_1_ofms_width = bottleneck_1_ofms_width;
const int chain_0_1_output_num_tiles_d = (bottleneck_1_ofms_depth / pw_tile_d)
		+ ((layer_5_pw_num_fils / pw_tile_d) != 0);
const int chain_0_1_output_num_tiles_h = (bottleneck_1_ofms_height / pw_tile_h)
		+ ((layer_5_pw_ofm_height % pw_tile_h) != 0);
const int chain_0_1_output_num_tiles_w = (bottleneck_1_ofms_width / pw_tile_w)
		+ ((layer_5_pw_ofm_width % pw_tile_w) != 0);

const int chain_0_1_max_filter_dim = bottleneck_1_dw_filter_dim;
const int chain_0_1_first_filter_dim = bottleneck_0_dw_filter_dim;
const int chain_0_1_max_strides = layer_0_s_strides;
const int chain_0_1_first_strides = layer_0_s_strides;
const int chain_0_1_first_padding_left = 0;
const int chain_0_1_max_rows_at_once =
		bottleneck_0_rows_at_once > bottleneck_1_rows_at_once ?
				bottleneck_0_rows_at_once : bottleneck_0_rows_at_once;
const int chain_0_1_in_buffer_height = chain_0_1_max_filter_dim
		+ (chain_0_1_max_rows_at_once - 1) * chain_0_1_max_strides;
const int chain_0_1_rows_filled_each_time = chain_0_1_max_rows_at_once
		* chain_0_1_max_strides;
const int chain_0_1_extra_cols_filled_first_time = chain_0_1_in_buffer_height
		- chain_0_1_rows_filled_each_time;
const int chain_0_1_extra_rows_filled_first_time = chain_0_1_in_buffer_height
		- chain_0_1_rows_filled_each_time;
const int chain_0_1_input_size = bottlenck_0_input_buffer_size;
const int chain_0_1_output_size = bottlenck_1_output_buffer_size;
const int chain_0_1_first_dw_layer_in_the_chain = 1;

const int chain_0_1_bottleneck_0_rows_at_once = 2;
const int chain_0_1_bottleneck_1_rows_at_once = 1;

void _0_1_bottlenecks_chain(
		fms_grp_dt channels[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		fms_dt result[max_fms_size]);

#endif
