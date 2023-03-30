
#include "../headers/conv_utils.h"

void padd_fms_tile_top_left(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                            fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                            fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                            fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                            fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                            fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                            const int starting_d,
                            const int tile_in_h,
                            const int tile_in_w,
                            const int padding_top_left,
                            const int ifms_d,
                            const int num_of_ifm_tiles_h,
                            const int num_of_ifm_tiles_w,
                            fms_dt fms_zero_point)
{
#pragma HLS INLINE

    const int num_of_tiles_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w;

    for (int o_d = 0; o_d < CHANNELS_PIPELINE_DEPTH; o_d++)
    {
#pragma HLS PIPELINE

        for (int i_d = 0; i_d < CHANNELS_TILE_DEPTH; i_d++)
        {
#pragma HLS UNROLL
            const int d = o_d * CHANNELS_PIPELINE_DEPTH + i_d;
            if (starting_d + d >= ifms_d)
            {
                break;
            }
            const int main_tile_index = (starting_d + d) * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_h + tile_in_w;
            const int top_tile_index = main_tile_index - num_of_ifm_tiles_w;
            const int left_tile_index = main_tile_index - 1;
            const int top_left_tile_index = top_tile_index - 1;
            const int top_right_tile_index = top_tile_index + 1;
            const int bottom_left_tile_index = num_of_ifm_tiles_w - 1;
            // top left corner
            for (int h = 0; h < MAX_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left || h >= padding_top_left)
                    {
                        break;
                    }
                    if (tile_in_h == 0 || tile_in_w == 0)
                    {
                        padding_top_left_buffer[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
                                               [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] = fms_zero_point;
                    }
                    else
                    {
                        padding_top_left_buffer[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
                                               [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] =
                                                   channels[top_left_tile_index][CHANNELS_TILE_HEIGHT - (padding_top_left - h)]
                                                           [CHANNELS_TILE_WIDTH - (padding_top_left - w)];
                    }
                }
            }
            // top right corner
            for (int h = 0; h < MAX_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left || h >= padding_top_left)
                    {
                        break;
                    }
                    if (tile_in_h == 0 || tile_in_w == num_of_ifm_tiles_w - 1)
                    {
                        padding_top_right_buffer[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
                                                [w] = fms_zero_point;
                    }
                    else
                    {
                        padding_top_right_buffer[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
                                                [w] =
                                                    channels[top_right_tile_index][CHANNELS_TILE_HEIGHT - (padding_top_left - h)]
                                                            [w];
                    }
                }
            }
            // bottom left corner
            for (int h = 0; h < MAX_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left || h >= padding_top_left)
                    {
                        break;
                    }
                    if (tile_in_h == num_of_ifm_tiles_h - 1 || tile_in_w == 0)
                    {
                        padding_bottom_left_buffer[d][h]
                                                  [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] = fms_zero_point;
                    }
                    else
                    {
                        padding_bottom_left_buffer[d][h]
                                                  [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] =
                                                      channels[top_left_tile_index][h]
                                                              [CHANNELS_TILE_HEIGHT - (padding_top_left - w)];
                    }
                }
            }
            // top
            for (int h = 0; h < MAX_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                if (h >= padding_top_left)
                {
                    break;
                }
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    padding_top_buffer[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)][w] =
                        channels[top_tile_index][CHANNELS_TILE_HEIGHT - (padding_top_left - h)][w];
                }
            }
            // left
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left)
                    {
                        break;
                    }
                    padding_left_buffer[d][h]
                                       [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] =
                                           channels[left_tile_index][h][CHANNELS_TILE_WIDTH - (padding_top_left - w)];
                }
            }
        }
    }
}

void padd_fms_tile_bottom_right(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                                fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                                fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                                fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                                const int starting_d,
                                const int tile_in_h,
                                const int tile_in_w,
                                const int padding_bottom_right,
                                const int ifms_d,
                                const int num_of_ifm_tiles_h,
                                const int num_of_ifm_tiles_w,
                                fms_dt fms_zero_point)
{
#pragma HLS INLINE

    const int num_of_tiles_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w;

    for (int o_d = 0; o_d < CHANNELS_PIPELINE_DEPTH; o_d++)
    {
#pragma HLS PIPELINE

        for (int i_d = 0; i_d < CHANNELS_TILE_DEPTH; i_d++)
        {
#pragma HLS UNROLL
            const int d = o_d * CHANNELS_PIPELINE_DEPTH + i_d;
            if (starting_d + d >= ifms_d)
            {
                break;
            }
            const int main_tile_index = (starting_d + d) * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_h + tile_in_w;
            const int bottom_tile_index = main_tile_index - num_of_ifm_tiles_w;
            const int right_tile_index = main_tile_index - 1;
            const int bottom_right_tile_index = bottom_tile_index + 1;
            // bottom right corner
            for (int h = 0; h < MAX_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_bottom_right || h >= padding_bottom_right)
                    {
                        break;
                    }
                    if (tile_in_h == num_of_ifm_tiles_h - 1 || tile_in_w == num_of_ifm_tiles_w - 1)
                    {
                        padding_bottom_right_buffer[d][h]
                                                   [w] = fms_zero_point;
                    }
                    else
                    {
                        padding_bottom_right_buffer[d][h][w] =
                            channels[bottom_right_tile_index][h][w];
                    }
                }
            }
            // bottom
            for (int h = 0; h < MAX_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                if (h >= padding_bottom_right)
                {
                    break;
                }
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    padding_bottom_buffer[d][h][w] =
                        channels[bottom_tile_index][h][w];
                }
            }
            // right
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_bottom_right)
                    {
                        break;
                    }
                    padding_right_buffer[d][h][w] =
                        channels[right_tile_index][h][w];
                }
            }
        }
    }
}

void fill_fms_tile(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                   fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                   fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                   fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                   fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                   fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                   fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                   fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                   fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                   fms_dt channels_tile[CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
                   conv_type layer_type,
                   const int starting_d,
                   const int tile_in_h,
                   const int tile_in_w,
                   const int padding_top_left,
                   const int ifms_d,
                   const int num_of_ifm_tiles_h,
                   const int num_of_ifm_tiles_w,
                   fms_dt fms_zero_point)
{
#pragma HLS INLINE

    // bottom right corner
    for (int h = 0; h < MAX_PADDING_BOTTOM_RIGHT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < MAX_PADDING_BOTTOM_RIGHT; w++)
        {
#pragma HLS UNROLL
            channels_tile[h + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_HEIGHT]
                         [w + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_WIDTH] =
                             padding_bottom_right_buffer[starting_d][h][w];
        }
    }
    // other corners
    for (int h = 0; h < MAX_PADDING_TOP_LEFT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < MAX_PADDING_TOP_LEFT; w++)
        {
#pragma HLS UNROLL
            channels_tile[h][w] = padding_top_left_buffer[starting_d][h][w];
            channels_tile[h][w + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_WIDTH] =
                padding_top_right_buffer[starting_d][h][w];
            channels_tile[h + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_HEIGHT][w] =
                padding_bottom_left_buffer[starting_d][h][w];
        }
    }
    // top and bottom
    for (int h = 0; h < MAX_PADDING_BOTTOM_RIGHT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
        {
#pragma HLS UNROLL
            channels_tile[h][w] = padding_top_buffer[starting_d][h][w + MAX_PADDING_TOP_LEFT];
            channels_tile[h][w] = padding_bottom_buffer[starting_d][h + CHANNELS_TILE_HEIGHT + MAX_PADDING_TOP_LEFT]
                                                       [w + MAX_PADDING_TOP_LEFT];
        }
    }
    // right and left
    for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < MAX_PADDING_BOTTOM_RIGHT; w++)
        {
#pragma HLS UNROLL
            channels_tile[h][w] = padding_left_buffer[starting_d][h + MAX_PADDING_TOP_LEFT][w];
            channels_tile[h][w] = padding_right_buffer[starting_d][h + MAX_PADDING_TOP_LEFT]
                                                      [w + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_WIDTH];
        }
    }

    for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
        {
#pragma HLS UNROLL
            channels_tile[h + MAX_PADDING_TOP_LEFT][w + MAX_PADDING_TOP_LEFT] =
                channels[starting_d][h][w];
        }
    }
}

void fill_fms_tile_dw(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                      fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                      fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                      fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                      fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                      fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                      fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                      fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                      fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                      fms_dt channels_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
                      conv_type layer_type,
                      const int starting_d,
                      const int tile_in_h,
                      const int tile_in_w,
                      const int padding_top_left,
                      const int ifms_d,
                      const int num_of_ifm_tiles_h,
                      const int num_of_ifm_tiles_w,
                      fms_dt fms_zero_point)
{
#pragma HLS INLINE

    for (int o_d = 0; o_d < CHANNELS_PIPELINE_DEPTH; o_d++)
    {
#pragma HLS PIPELINE

        for (int i_d = 0; i_d < CHANNELS_TILE_DEPTH; i_d++)
        {
#pragma HLS UNROLL
            const int d = o_d * CHANNELS_PIPELINE_DEPTH + i_d;
            if (starting_d + d >= ifms_d)
            {
                break;
            }
            // bottom right corner
            for (int h = 0; h < MAX_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    channels_tile[d][h + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_HEIGHT]
                                 [w + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_WIDTH] =
                                     padding_bottom_right_buffer[d][h][w];
                }
            }
            // other corners
            for (int h = 0; h < MAX_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    channels_tile[d][h][w] = padding_top_left_buffer[d][h][w];
                    channels_tile[d][h][w + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_WIDTH] =
                        padding_top_right_buffer[d][h][w];
                    channels_tile[d][h + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_HEIGHT][w] =
                        padding_bottom_left_buffer[d][h][w];
                }
            }
            // top and bottom
            for (int h = 0; h < MAX_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    channels_tile[d][h][w] = padding_top_buffer[d][h][w + MAX_PADDING_TOP_LEFT];
                    channels_tile[d][h][w] = padding_bottom_buffer[d][h + CHANNELS_TILE_HEIGHT + MAX_PADDING_TOP_LEFT]
                                                                  [w + MAX_PADDING_TOP_LEFT];
                }
            }
            // right and left
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    channels_tile[d][h][w] = padding_left_buffer[d][h + MAX_PADDING_TOP_LEFT][w];
                    channels_tile[d][h][w] = padding_right_buffer[d][h + MAX_PADDING_TOP_LEFT]
                                                                 [w + MAX_PADDING_TOP_LEFT + CHANNELS_TILE_WIDTH];
                }
            }

            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    channels_tile[d][h + MAX_PADDING_TOP_LEFT][w + MAX_PADDING_TOP_LEFT] =
                        channels[d][h][w];
                }
            }
        }
    }
}

void copy_fms_tile_corners(fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                           fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                           fms_dt padding_top_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_left_buffer_dst[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_top_left_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_top_right_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_left_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_TOP_LEFT][MAX_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_right_buffer_dst[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_PADDING_BOTTOM_RIGHT],
                           fms_dt padding_bottom_right_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_PADDING_BOTTOM_RIGHT][MAX_PADDING_BOTTOM_RIGHT],
                           const int starting_d,
                           const int ifms_d,
                           const int padding_top_left,
                           const int padding_bottom_right)
{
#pragma HLS INLINE off

    for (int o_d = 0; o_d < CHANNELS_PIPELINE_DEPTH; o_d++)
    {
#pragma HLS PIPELINE

        for (int i_d = 0; i_d < CHANNELS_TILE_DEPTH; i_d++)
        {
#pragma HLS UNROLL
            const int d = o_d * CHANNELS_PIPELINE_DEPTH + i_d;
            if (starting_d + d >= ifms_d)
            {
                break;
            }
            // bottom right corner
            for (int h = 0; h < MAX_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    padding_bottom_right_buffer_dst[d][h][w] =
                        padding_bottom_right_buffer[d][h][w];
                }
            }
            // other corners
            for (int h = 0; h < MAX_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    padding_top_left_buffer_dst[d][h][w] =
                        padding_top_left_buffer[d][h][w];
                    padding_top_right_buffer_dst[d][h][w] =
                        padding_top_right_buffer[d][h][w];
                    padding_bottom_left_buffer_dst[d][h][w] =
                        padding_bottom_left_buffer[d][h][w];
                }
            }

            // top and bottom
            for (int h = 0; h < MAX_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    padding_top_buffer_dst[d][h][w] = padding_top_buffer[d][h][w];
                    padding_bottom_buffer_dst[d][h][w] = padding_bottom_buffer[d][h][w];
                }
            }
            // right and left
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    padding_right_buffer_dst[d][h][w] = padding_right_buffer[d][h][w];
                    padding_left_buffer_dst[d][h][w] = padding_left_buffer[d][h][w];
                }
            }
        }
    }
}

void normalize_and_write_back_result_tile(fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                                          dw_pss_dt result_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                                          fms_quantization_scheme normalization, const int layer_relu,
                                          const fused_scales_dt fused_scales_tile[],
                                          const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
                                          const relu_6_fused_scales_dt relu_6_fused_scales_tile[],
                                          const biases_dt fused_zero_points_tile[],
                                          const int ifms_d,
                                          const int tile_in_h,
                                          const int tile_in_w,
                                          const int starting_d,
                                          const int num_of_ofm_tiles_h,
                                          const int num_of_ofm_tiles_w)
{
#pragma HLS INLINE off

    const int num_of_tiles_hw = num_of_ofm_tiles_h * num_of_ofm_tiles_w;

    for (int o_d = starting_d; o_d < CHANNELS_PIPELINE_DEPTH; o_d++)
    {
#pragma HLS PIPELINE
        for (int i_d = 0; i_d < CHANNELS_TILE_DEPTH; i_d++)
        {
#pragma HLS UNROLL
            const int d = o_d * CHANNELS_PIPELINE_DEPTH + i_d;
            const int main_tile_index = (starting_d + d) * num_of_tiles_hw + tile_in_h * num_of_ofm_tiles_w + tile_in_w;
            if (starting_d + d >= ifms_d)
            {
                break;
            }
            normalization.fused_scales = fused_scales_tile[d];
            normalization.fused_scales_log_2_shift =
                fused_scales_log_2_shifts_tile[d];
            normalization.relu_6_fused_scale =
                relu_6_fused_scales_tile[d];
            normalization.fused_zero_point =
                fused_zero_points_tile[d];
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    result[main_tile_index][h][w] = dw_relu_norm(
                        result_tile[d][h][w], normalization,
                        layer_relu);
                }
            }
        }
    }
}
