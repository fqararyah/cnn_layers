#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"
#include "../headers/conv_utils.h"

void dw_conv_engine(
		dw_weights_dt weights[CHANNELS_PIPELINE_DEPTH][max_filter_hw_dim * max_filter_hw_dim],
		fms_dt ifms_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
		dw_pss_dt result_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
		const int filter_dim, const int strides,
        const int layer_d,
        const int starting_d) {
#pragma HLS INLINE off

	dw_conv_engine: for (int c_h = 0; c_h < max_filter_hw_dim; c_h++) {
		for (int c_w = 0; c_w < max_filter_hw_dim; c_w++) {
			for (int d_in_pipeline = 0; d_in_pipeline < CHANNELS_PIPELINE_DEPTH;
					d_in_pipeline++) {
#pragma HLS PIPELINE
                if(starting_d + d_in_pipeline >= layer_d){
                    break;
                }
				for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++) {
#pragma HLS UNROLL
					for (int w = 0; w < CHANNELS_TILE_WIDTH; w++) {
#pragma HLS UNROLL
						if (c_w >= filter_dim || c_h >= filter_dim
								|| h >= dw_tile_h / strides
								|| w >= dw_tile_w / strides) {
							break;
						}
						if (c_h == 0 && c_w == 0) {
							result_tile[d_in_pipeline][h][w] =
									ifms_buffer[d_in_pipeline][h * strides + c_h][w
											* strides + c_w]
											* weights[d_in_pipeline][c_h
													* filter_dim + c_w];
						} else {
							result_tile[d_in_pipeline][h][w] +=
									ifms_buffer[d_in_pipeline][h * strides + c_h][w
											* strides + c_w]
											* weights[d_in_pipeline][c_h
													* filter_dim + c_w];
						}
					}
				}
			}
		}
	}
}

void dw_conv_3x3(const dw_weights_dt weights[][3 * 3],
                    fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                    fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                    const int layer, const int layer_conv_d, const int layer_width, const int layer_height,
                    const int num_of_tiles_d,
                    const int num_of_ifms_tiles_h, const int num_of_ifms_tiles_w,
                    const int num_of_tiles_h, const int num_of_tiles_w,
                    const int strides, const int padding_left, const int padding_right, const int padding_top,
                    const fused_scales_dt fused_scales[],
                    const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                    const relu_6_fused_scales_dt relu_6_fused_scales[], const biases_dt fused_zero_points[],
                    const fused_scales_dt fused_scales_part2[],
                    const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
                    const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
                    const biases_dt fused_zero_points_part2[])
{
    #pragma HLS INLINE off

}