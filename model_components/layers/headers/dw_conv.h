#ifndef DW_CONV_H
#define DW_CONV_H

#include "../../basic_defs/basic_defs_glue.h"

namespace seml_engines
{
	void fill_layer_dw_weights_off_chip(const dw_weights_dt *dw_weights,
										dw_weights_dt layer_dw_weights[][3 * 3],
										const int current_dw_layer_weights_offset,
										const int layer_depth);
	void fill_dw_weights_tile(const dw_weights_dt weights[][3 * 3],
							  dw_weights_dt weights_tile[][3 * 3],
							  int starting_d, const int current_dw_layer_weights_offset);

	void dw_conv_3x3(const dw_weights_dt weights[][3 * 3],
					 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
					 const int layer,
					 const layer_specs layer_specs_struct,
					 const fused_scales_dt fused_scales[],
					 const relu_6_fused_scales_dt relu_6_fused_scales[], const biases_dt fused_zero_points[]);

	// V2
	void dw_conv_3x3(const dw_weights_dt weights[][3 * 3],
					 fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
					 fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
					 const int layer,
					 const layer_specs layer_specs_struct,
					 const fused_scales_dt fused_scales[],
					 const relu_6_fused_scales_dt relu_6_fused_scales[], const biases_dt fused_zero_points[],
					 const int model_configs_list[2 * max_conv_layers],
					 const int to_produce_row_count,
					 const int starting_row = 0);

	void dw_conv_5x5(dw_weights_dt weights[max_conv_d][5][5],
					 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
					 const int layer, const int layer_conv_d, const int num_of_tiles_d,
					 const int num_of_tiles_h, const int num_of_tiles_w, const int strides);

	void dw_conv_7x7(dw_weights_dt weights[max_conv_d][7][7],
					 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
					 const int layer, const int layer_conv_d, const int num_of_tiles_d,
					 const int num_of_tiles_h, const int num_of_tiles_w, const int strides);
}
#endif
