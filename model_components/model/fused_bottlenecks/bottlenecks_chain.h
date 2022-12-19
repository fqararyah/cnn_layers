#ifndef BOTTLENECKS_CHAIN
#define BOTTLENECKS_CHAIN
#include "bottlenecks_parallelism.h"
#include "bottleneck_kernels.h"
#include "bottleneck.h"

struct bottlenecks_chain_specs {
	int chain_ifms_depth;
	int chain_ifms_height;
	int chain_ifms_width;
	int chain_ofms_depth;
	int chain_ofms_height;
	int chain_ofms_width;
	int chain_output_num_tiles_d;
	int chain_output_num_tiles_h;
	int chain_output_num_tiles_w;

	int chain_max_filter_dim;
	int first_filter_dim;
	int chain_max_strides;
	int first_strides;
	int first_dw_padding_left;
	int first_dw_padding_right;
	int first_dw_padding_top;
	int first_dw_padding_bottom;
	int last_dw_padding_left;
	int last_dw_padding_right;
	int last_dw_padding_top;
	int last_dw_padding_bottom;
	int chain_max_rows_at_once;
	int chain_input_height;
	int chain_input_size;
	int chain_output_size;
	int first_dw_layer_in_the_chain;
};

const bottlenecks_chain_specs _1_chain_specs = { bottleneck_1_ifms_depth,
		bottleneck_1_ifms_height, bottleneck_1_ifms_width,
		bottleneck_1_ofms_depth, bottleneck_1_ofms_height,
		bottleneck_1_ofms_width, (bottleneck_1_ofms_depth / pw_tile_d)
				+ ((layer_6_pw_num_fils / pw_tile_d) != 0),
		(bottleneck_1_ofms_height / pw_tile_h)
				+ ((layer_6_pw_ofm_height % pw_tile_h) != 0),
		(bottleneck_1_ofms_width / pw_tile_w)
				+ ((layer_6_pw_ofm_width % pw_tile_w) != 0),

		bottleneck_1_dw_filter_dim, bottleneck_1_dw_filter_dim,
		bottleneck_1_dw_strides, bottleneck_1_dw_strides,
		bottleneck_1_dw_padding_left, bottleneck_1_dw_padding_right,
		bottleneck_1_dw_padding_top, bottleneck_1_dw_padding_bottom,
		bottleneck_1_dw_padding_left, bottleneck_1_dw_padding_right,
		bottleneck_1_dw_padding_top, bottleneck_1_dw_padding_bottom, 1, 2, // chain_max_rows_at_once * chain_max_strides
		bottlenck_1_input_buffer_size, bottlenck_1_output_buffer_size, 5 };

void _1_bottlenecks_chain(
		fms_dt chain_input[], // chain_input_height*chain_input_width*chain_input_depth
		fms_dt result[max_fms_size], const bottlenecks_chain_specs chain_specs,
		int starting_h, int filling_row);

#endif
