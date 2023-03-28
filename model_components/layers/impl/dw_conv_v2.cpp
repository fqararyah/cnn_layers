#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"
#include "../headers/pw_conv.h"

void dw_conv_3x3_v2(const dw_weights_dt weights[][3 * 3],
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
}