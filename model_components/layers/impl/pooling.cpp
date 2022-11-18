#include "../headers/layers_imp_common_includes.h"
#include "../headers/pooling.h"

void avgpool(fms_dt channels[max_fms_size],
		fms_dt result[fc_layer_input_size]) {
#pragma HLS INLINE OFF
	const int avgpool_input_hw = avgpool_input_height * avgpool_input_width;

	const int num_tiles_h =
			(avgpool_input_height % pw_tile_h == 0) ?
					avgpool_input_height / pw_tile_h :
					1 + avgpool_input_height / pw_tile_h;
	const int num_tiles_w =
			(avgpool_input_width % pw_tile_w == 0) ?
					avgpool_input_width / pw_tile_w :
					1 + avgpool_input_width / pw_tile_w;
	const int num_tiles_hw = num_tiles_h * num_tiles_w;

	dw_conv_itd_loop: for (int d = 0; d < avgpool_input_depth; d++) {
		pss_dt tmp = 0;
		const int tile_in_d = (d / pw_tile_d);
		const int in_tile_d = (d % pw_tile_d);
		dw_conv_ith_loop: for (int h = 0; h < avgpool_input_height; h++) {
			const int tile_in_h = h / pw_tile_h;
			const int in_tile_h = h % pw_tile_h;
			dw_conv_itw_loop: for (int w = 0; w < avgpool_input_width; w++) {
				const int tile_in_w = w / pw_tile_w;
				const int in_tile_w = w % pw_tile_w;
				const int tile_index = tile_in_d * num_tiles_hw
						+ tile_in_h * num_tiles_w + tile_in_w;
				const int in_tile_index = in_tile_d * pw_tile_hw
						+ in_tile_h * pw_tile_w + in_tile_w;
				tmp += channels[tile_index * pw_tile_size + in_tile_index];
			}
		}
		pss_f_dt scaled_tmp = (((pss_f_dt) tmp) / avgpool_input_hw
				- pooling_ifms_zero_point) * pooling_fused_scale
				+ pooling_ofms_zero_point;

		clamp(scaled_tmp);

		result[d] = (fms_dt) (scaled_tmp);
	}
}
