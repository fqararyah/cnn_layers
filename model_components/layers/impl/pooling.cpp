#include "../headers/layers_imp_common_includes.h"
#include "../headers/pooling.h"

void avgpool(fms_dt channels[][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
			 fms_dt result[fc_layer_input_size],
			 const pooling_layer_specs layer_specs_struct)
{

	const int avgpool_input_depth = layer_specs_struct.ifm_depth;
	const int avgpool_input_height = layer_specs_struct.ifm_height;
	const int avgpool_input_width = layer_specs_struct.ifm_width;
	const int avgpool_input_hw = avgpool_input_height * avgpool_input_width;
	for (int d = 0; d < avgpool_input_depth; d++)
	{
		pss_dt tmp = 0;
	avgpool_ith_loop:
		for (int h = 0; h < avgpool_input_height; h++)
		{
		avgpool_itw_loop:
			for (int w = 0; w < avgpool_input_width; w++)
			{
				tmp += channels[d][h][w];
			}
		}
		pss_f_dt scaled_tmp = (tmp / avgpool_input_hw - layer_specs_struct.ifms_zero_point) *
								  layer_specs_struct.fused_scale +
							  layer_specs_struct.ofms_zero_point;

		result[d] = clamp(scaled_tmp);
	}
}

void avgpool(fms_dt channels[max_fms_size],
			 fms_dt result[fc_layer_input_size],
			 const pooling_layer_specs layer_specs_struct)
{
#pragma HLS INLINE OFF
	
	const int avgpool_input_depth = layer_specs_struct.ifm_depth;
	const int avgpool_input_height = layer_specs_struct.ifm_height;
	const int avgpool_input_width = layer_specs_struct.ifm_width;
	const int avgpool_input_hw = avgpool_input_height * avgpool_input_width;

	const int num_tiles_h =
		(avgpool_input_height % pw_tile_h == 0) ? avgpool_input_height / pw_tile_h : 1 + avgpool_input_height / pw_tile_h;
	const int num_tiles_w =
		(avgpool_input_width % pw_tile_w == 0) ? avgpool_input_width / pw_tile_w : 1 + avgpool_input_width / pw_tile_w;
	const int num_tiles_hw = num_tiles_h * num_tiles_w;

avgpool_itd_loop:
	for (int d = 0; d < avgpool_input_depth; d++)
	{
		pss_dt tmp = 0;
		const int tile_in_d = (d / pw_tile_d);
		const int in_tile_d = (d % pw_tile_d);
	avgpool_ith_loop:
		for (int h = 0; h < avgpool_input_height; h++)
		{
			const int tile_in_h = h / pw_tile_h;
			const int in_tile_h = h % pw_tile_h;
		avgpool_itw_loop:
			for (int w = 0; w < avgpool_input_width; w++)
			{
				const int tile_in_w = w / pw_tile_w;
				const int in_tile_w = w % pw_tile_w;
				const int tile_index = tile_in_d * num_tiles_hw + tile_in_h * num_tiles_w + tile_in_w;
				const int in_tile_index = in_tile_d * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;
				tmp += channels[tile_index * pw_tile_size + in_tile_index];
			}
		}
		pss_f_dt scaled_tmp = (tmp / avgpool_input_hw - layer_specs_struct.ifms_zero_point) *
								  layer_specs_struct.fused_scale +
							  layer_specs_struct.ofms_zero_point;

		result[d] = clamp(scaled_tmp);
	}
}
