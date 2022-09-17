#include "../headers/pooling.h"

void avgpool(fms_dt channels[max_fms_size],
		fms_dt result[fc_layer_input_size]) {
#pragma HLS INLINE OFF
	const int avgpool_input_hw = avgpool_input_height * avgpool_input_width;

	dw_conv_itd_loop: for (int d = 0; d < avgpool_input_depth; d++) {
		int tmp;
		dw_conv_ith_loop: for (int h = 0; h < avgpool_input_height; h++) {
			dw_conv_itw_loop: for (int w = 0; w < avgpool_input_width; w++) {
				tmp += channels[d * pw_tile_hw + h * pw_tile_w + w];
			}
		}
		result[d] = (fms_dt) (tmp / avgpool_input_hw);
	}

}
