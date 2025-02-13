
#include "../headers/conv_utils.h"

void seml_engines::fill_fused_scales(const fused_scales_dt *off_chip_fused_scales,
                                     fused_scales_dt layer_fused_scales[],
                                     const int current_dw_layer_fused_scales_offset,
                                     const int layer_num_filters)
{
#pragma HLS INLINE off

    for (int i = 0; i < max_num_filters; i++)
    {
#pragma HLS PIPELINE II = 1
        if (i >= layer_num_filters)
        {
            break;
        }
        layer_fused_scales[i] = off_chip_fused_scales[current_dw_layer_fused_scales_offset + i];
    }
}

void seml_engines::fill_fused_zero_points(const biases_dt *off_chip_fused_zero_points,
                                          biases_dt layer_fused_zero_points[],
                                          const int current_dw_layer_fused_zps_offset,
                                          const int layer_num_filters)
{
#pragma HLS INLINE off

    for (int i = 0; i < max_num_filters; i++)
    {
#pragma HLS PIPELINE II = 1

        if (i >= layer_num_filters)
        {
            break;
        }
        layer_fused_zero_points[i] = off_chip_fused_zero_points[current_dw_layer_fused_zps_offset + i];
    }
}

void padd_fms_tile_top_left(fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                            fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                            fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_TOP_LEFT],
                            fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                            fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                            fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                            const int starting_d,
                            const int tile_in_h,
                            const int tile_in_w,
                            const int padding_top_left,
                            const int ifms_d,
                            const int num_of_ifm_tiles_h,
                            const int num_of_ifm_tiles_w,
                            const int ifms_height,
                            const int ifms_width,
                            fms_dt fms_zero_point)
{
#pragma HLS INLINE off

    const int num_of_tiles_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w;

    for (int o_d = 0; o_d < CHANNELS_PIPELINE_DEPTH; o_d += CHANNELS_TILE_DEPTH)
    {
#pragma HLS PIPELINE

        for (int i_d = 0; i_d < CHANNELS_TILE_DEPTH; i_d++)
        {
#pragma HLS UNROLL
            const int d = o_d * CHANNELS_TILE_DEPTH + i_d;
            if (starting_d + d >= ifms_d)
            {
                break;
            }
            const int main_tile_index = (starting_d + d) * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_h + tile_in_w;
            const int top_tile_index = main_tile_index - num_of_ifm_tiles_w;
            const int left_tile_index = main_tile_index - 1;
            const int top_left_tile_index = top_tile_index - 1;
            const int top_right_tile_index = top_tile_index + 1;
            const int bottom_left_tile_index = main_tile_index + num_of_ifm_tiles_w - 1;
            // top left corner
            for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left || h >= padding_top_left)
                    {
                        break;
                    }
                    if (tile_in_h == 0 || tile_in_w == 0)
                    {
                        padding_top_left_buffer[d][h][w] = fms_zero_point;
                    }
                    else
                    {
                        padding_top_left_buffer[d][h][w] =
                            channels[top_left_tile_index][CHANNELS_TILE_HEIGHT - (padding_top_left - h)]
                                    [CHANNELS_TILE_WIDTH - (padding_top_left - w)];
                    }
                }
            }
            // top right corner
            for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left || h >= padding_top_left)
                    {
                        break;
                    }
                    if (tile_in_h == 0 || tile_in_w == num_of_ifm_tiles_w - 1)
                    {
                        padding_top_right_buffer[d][h]
                                                [w] = fms_zero_point;
                    }
                    else
                    {
                        padding_top_right_buffer[d][h][w] =
                            channels[top_right_tile_index][CHANNELS_TILE_HEIGHT - (padding_top_left - h)]
                                    [w];
                    }
                }
            }
            // bottom left corner
            for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left || h >= padding_top_left)
                    {
                        break;
                    }
                    if (tile_in_h == num_of_ifm_tiles_h - 1 || tile_in_w == 0)
                    {
                        padding_bottom_left_buffer[d][h][w] = fms_zero_point;
                    }
                    else
                    {
                        padding_bottom_left_buffer[d][h][w] =
                            channels[bottom_left_tile_index][h]
                                    [CHANNELS_TILE_WIDTH - (padding_top_left - w)];
                    }
                }
            }
            // top
            for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                if (h >= padding_top_left)
                {
                    break;
                }
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    if (tile_in_h == 0 || tile_in_w * CHANNELS_TILE_WIDTH + w >= ifms_width)
                    {
                        padding_top_buffer[d][h][w] = fms_zero_point;
                    }
                    else
                    {
                        padding_top_buffer[d][h][w] =
                            channels[top_tile_index][CHANNELS_TILE_HEIGHT - (padding_top_left - h)][w];
                    }
                }
            }
            // left
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left)
                    {
                        break;
                    }
                    if (tile_in_w == 0 || tile_in_h * CHANNELS_TILE_HEIGHT + h >= ifms_height)
                    {
                        padding_left_buffer[d][h][w] = fms_zero_point;
                    }
                    else
                    {
                        padding_left_buffer[d][h][w] =
                            channels[left_tile_index][h][CHANNELS_TILE_WIDTH - (padding_top_left - w)];
                    }
                }
            }
        }
    }
}

void padd_fms_tile_bottom_right(fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                                fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                                fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
                                fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
                                const int starting_d,
                                const int tile_in_h,
                                const int tile_in_w,
                                const int padding_bottom_right,
                                const int ifms_d,
                                const int num_of_ifm_tiles_h,
                                const int num_of_ifm_tiles_w,
                                const int ifms_height,
                                const int ifms_width,
                                fms_dt fms_zero_point)
{
#pragma HLS INLINE off

    const int num_of_tiles_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w;

    for (int o_d = 0; o_d < CHANNELS_PIPELINE_DEPTH; o_d += CHANNELS_TILE_DEPTH)
    {
#pragma HLS PIPELINE

        for (int i_d = 0; i_d < CHANNELS_TILE_DEPTH; i_d++)
        {
#pragma HLS UNROLL
            const int d = o_d * CHANNELS_TILE_DEPTH + i_d;
            if (starting_d + d >= ifms_d)
            {
                break;
            }
            const int main_tile_index = (starting_d + d) * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_w + tile_in_w;
            const int bottom_tile_index = main_tile_index + num_of_ifm_tiles_w;
            const int right_tile_index = main_tile_index + 1;
            const int bottom_right_tile_index = bottom_tile_index + 1;
            // bottom right corner
            for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_bottom_right || h >= padding_bottom_right)
                    {
                        break;
                    }
                    if (tile_in_h == num_of_ifm_tiles_h - 1 || tile_in_w == num_of_ifm_tiles_w - 1)
                    {
                        padding_bottom_right_buffer[d][h][w] = fms_zero_point;
                    }
                    else
                    {
                        padding_bottom_right_buffer[d][h][w] =
                            channels[bottom_right_tile_index][h][w];
                    }
                }
            }
            // bottom
            for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                if (h >= padding_bottom_right)
                {
                    break;
                }
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    if (tile_in_h == num_of_ifm_tiles_h - 1 || tile_in_w * CHANNELS_TILE_WIDTH + w >= ifms_width)
                    {
                        padding_bottom_buffer[d][h][w] = fms_zero_point;
                    }
                    else
                    {
                        padding_bottom_buffer[d][h][w] =
                            channels[bottom_tile_index][h][w];
                    }
                }
            }
            // right
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_bottom_right)
                    {
                        break;
                    }
                    if (tile_in_w == num_of_ifm_tiles_w - 1 || tile_in_h * CHANNELS_TILE_HEIGHT + h >= ifms_height)
                    {
                        padding_right_buffer[d][h][w] = fms_zero_point;
                    }
                    else
                    {
                        padding_right_buffer[d][h][w] =
                            channels[right_tile_index][h][w];
                    }
                }
            }
        }
    }
}

void fill_fms_tile(fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                   fms_dt channels_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
                   const int tile_in_d,
                   const int tile_in_h,
                   const int tile_in_w,
                   const fms_dt current_layer_zero_point,
                   const layer_specs layer_specs_struct,
                   const int model_configs_list_limit,
                   const int starting_row)
{
#pragma HLS INLINE off

    if (starting_row == 0)
    {
        fill_fms_tile(channels,
                      channels_tile,
                      tile_in_d,
                      tile_in_h,
                      tile_in_w,
                      current_layer_zero_point,
                      layer_specs_struct,
                      model_configs_list_limit);
    }
    else
    {
        const int num_of_ifm_tiles_h = layer_specs_struct.layer_num_of_ifm_tiles_h;
        const int num_of_ifm_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_h;
        const int strides = layer_specs_struct.strides;
        const int filter_dim = layer_specs_struct.filter_size;
        const int padding_bottom = filter_dim - strides;
        const int padding_top = layer_specs_struct.padding_top;
        const int padding_right = layer_specs_struct.padding_right;
        const int padding_left = layer_specs_struct.padding_left;
        const int layer_ifm_height = layer_specs_struct.layer_ifm_height;
        const int layer_ifm_width = layer_specs_struct.layer_ifm_width;
        const int layer_ifms_depth = layer_specs_struct.layer_depth;

        const int abs_ifm_h = starting_row * strides + tile_in_h * CHANNELS_TILE_HEIGHT - padding_top;

        const int num_of_tiles_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w;
        for (int d = 0; d < CHANNELS_PIPELINE_DEPTH; d++)
        {
#pragma HLS PIPELINE

            if (tile_in_d + d >= layer_ifms_depth || (model_configs_list_limit != 0 && tile_in_d + d >= model_configs_list_limit))
            {
                break;
            }

            const int main_tile_index = (tile_in_d + d) * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_w + tile_in_w;
            const int bottom_tile_index = main_tile_index + num_of_ifm_tiles_w;
            const int right_tile_index = main_tile_index + 1;
            const int bottom_right_tile_index = bottom_tile_index + 1;
            const int top_tile_index = main_tile_index - num_of_ifm_tiles_w;
            const int left_tile_index = main_tile_index - 1;
            const int top_left_tile_index = top_tile_index - 1;
            const int top_right_tile_index = top_tile_index + 1;
            const int bottom_left_tile_index = main_tile_index + num_of_ifm_tiles_w - 1;

            if (layer_specs_struct.conv_layer_type != PW_CONV)
            {
                // bottom right corner
                for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT_V2; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
                    {
#pragma HLS UNROLL
                        if (w >= padding_right || h >= padding_bottom)
                        {
                            break;
                        }
                        if (abs_ifm_h + CHANNELS_TILE_HEIGHT + h >= layer_ifm_height || tile_in_w == num_of_ifm_tiles_w - 1)
                        {
                            channels_tile[d][h + CHANNELS_TILE_HEIGHT]
                                         [w + padding_left + CHANNELS_TILE_WIDTH] = current_layer_zero_point;
                        }
                        else
                        {
                            channels_tile[d][h + CHANNELS_TILE_HEIGHT]
                                         [w + padding_left + CHANNELS_TILE_WIDTH] =
                                             channels[bottom_right_tile_index][h][w];
                        }
                    }
                }
                // bottom left corner
                for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT_V2; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
                    {
#pragma HLS UNROLL
                        if (w >= padding_left || h >= padding_bottom)
                        {
                            break;
                        }

                        if (abs_ifm_h + CHANNELS_TILE_HEIGHT + h >= layer_ifm_height || tile_in_w == 0)
                        {
                            channels_tile[d][h + CHANNELS_TILE_HEIGHT][w] = current_layer_zero_point;
                        }
                        else
                        {
                            channels_tile[d][h + CHANNELS_TILE_HEIGHT][w] =
                                channels[bottom_left_tile_index][h]
                                        [CHANNELS_TILE_WIDTH - (padding_left - w)];
                        }
                    }
                }
                // bottom

                for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT_V2; h++)
                {
#pragma HLS UNROLL
                    if (h >= padding_bottom)
                    {
                        break;
                    }
                    for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                    {
#pragma HLS UNROLL
                        if (abs_ifm_h + CHANNELS_TILE_HEIGHT + h >= layer_ifm_height || tile_in_w * CHANNELS_TILE_WIDTH + w >= layer_ifm_width)
                        {
                            channels_tile[d][h + CHANNELS_TILE_HEIGHT][w + padding_left] = current_layer_zero_point;
                        }
                        else
                        {
                            channels_tile[d][h + CHANNELS_TILE_HEIGHT][w + padding_left] =
                                channels[bottom_tile_index][h][w];
                        }
                    }
                }

                // left and right
                for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
                    {
#pragma HLS UNROLL
                        if (w >= padding_left)
                        {
                            break;
                        }
                        if (tile_in_w == 0 || abs_ifm_h + h >= layer_ifm_height)
                        {
                            channels_tile[d][h][w] = current_layer_zero_point;
                        }
                        else
                        {
                            channels_tile[d][h][w] =
                                channels[left_tile_index][h][CHANNELS_TILE_WIDTH - (padding_left - w)];
                        }
                    }
                }

                for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
                    {
#pragma HLS UNROLL
                        if (w >= padding_right)
                        {
                            break;
                        }
                        if (tile_in_w == num_of_ifm_tiles_w - 1 || abs_ifm_h + h >= layer_ifm_height)
                        {
                            channels_tile[d][h][w + padding_left + CHANNELS_TILE_WIDTH] = current_layer_zero_point;
                        }
                        else
                        {
                            channels_tile[d][h][w + padding_left + CHANNELS_TILE_WIDTH] =
                                channels[right_tile_index][h][w];
                        }
                    }
                }
            }
            // fill body
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    if (abs_ifm_h + h >= layer_ifm_height || tile_in_w * CHANNELS_TILE_WIDTH + w >= layer_ifm_width)
                    {
                        channels_tile[d][h][w + padding_left] = current_layer_zero_point;
                    }
                    else
                    {
                        channels_tile[d][h][w + padding_left] =
                            channels[main_tile_index][h][w];
                    }
                }
            }
        }
    }
}

void fill_fms_tile(fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                   fms_dt channels_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
                   const int tile_in_d,
                   const int tile_in_h,
                   const int tile_in_w,
                   const fms_dt current_layer_zero_point,
                   const layer_specs layer_specs_struct,
                   const int model_configs_list_limit)
{
#pragma HLS INLINE off

    const int num_of_ifm_tiles_h = layer_specs_struct.layer_num_of_ifm_tiles_h;
    const int num_of_ifm_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_h;
    const int padding_bottom_right = layer_specs_struct.padding_bottom;
    const int padding_top_left = layer_specs_struct.padding_top;
    const int layer_ifm_height = layer_specs_struct.layer_ifm_height;
    const int layer_ifm_width = layer_specs_struct.layer_ifm_width;
    const int layer_ifms_depth = layer_specs_struct.layer_depth;

    const int num_of_tiles_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w;

    if (layer_specs_struct.conv_layer_type != PW_CONV)
    {
        for (int d = 0; d < CHANNELS_PIPELINE_DEPTH; d++)
        {
#pragma HLS PIPELINE

            if (tile_in_d + d >= layer_ifms_depth || (model_configs_list_limit != 0 && tile_in_d + d >= model_configs_list_limit))
            {
                break;
            }

            const int main_tile_index = (tile_in_d + d) * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_w + tile_in_w;
            const int bottom_tile_index = main_tile_index + num_of_ifm_tiles_w;
            const int right_tile_index = main_tile_index + 1;
            const int bottom_right_tile_index = bottom_tile_index + 1;
            const int top_tile_index = main_tile_index - num_of_ifm_tiles_w;
            const int left_tile_index = main_tile_index - 1;
            const int top_left_tile_index = top_tile_index - 1;
            const int top_right_tile_index = top_tile_index + 1;
            const int bottom_left_tile_index = main_tile_index + num_of_ifm_tiles_w - 1;

            // bottom right corner
            for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_bottom_right || h >= padding_bottom_right)
                    {
                        break;
                    }
                    if (tile_in_h == num_of_ifm_tiles_h - 1 || tile_in_w == num_of_ifm_tiles_w - 1)
                    {
                        channels_tile[d][h + padding_top_left + CHANNELS_TILE_HEIGHT]
                                     [w + padding_top_left + CHANNELS_TILE_WIDTH] = current_layer_zero_point;
                    }
                    else
                    {
                        channels_tile[d][h + padding_top_left + CHANNELS_TILE_HEIGHT]
                                     [w + padding_top_left + CHANNELS_TILE_WIDTH] =
                                         channels[bottom_right_tile_index][h][w];
                    }
                }
            }
            // other corners
            for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left || h >= padding_top_left)
                    {
                        break;
                    }

                    if (tile_in_h == 0 || tile_in_w == 0)
                    {
                        channels_tile[d][h][w] = current_layer_zero_point;
                    }
                    else
                    {
                        channels_tile[d][h][w] =
                            channels[top_left_tile_index][CHANNELS_TILE_HEIGHT - (padding_top_left - h)]
                                    [CHANNELS_TILE_WIDTH - (padding_top_left - w)];
                    }

                    if (tile_in_h == 0 || tile_in_w == num_of_ifm_tiles_w - 1)
                    {
                        channels_tile[d][h][w + padding_top_left + CHANNELS_TILE_WIDTH] = current_layer_zero_point;
                    }
                    else
                    {
                        channels_tile[d][h][w + padding_top_left + CHANNELS_TILE_WIDTH] =
                            channels[top_right_tile_index][CHANNELS_TILE_HEIGHT - (padding_top_left - h)]
                                    [w];
                    }

                    if (tile_in_h == num_of_ifm_tiles_h - 1 || tile_in_w == 0)
                    {
                        channels_tile[d][h + padding_top_left + CHANNELS_TILE_HEIGHT][w] = current_layer_zero_point;
                    }
                    else
                    {
                        channels_tile[d][h + padding_top_left + CHANNELS_TILE_HEIGHT][w] =
                            channels[bottom_left_tile_index][h]
                                    [CHANNELS_TILE_WIDTH - (padding_top_left - w)];
                    }
                }
            }
        }

        for (int d = 0; d < CHANNELS_PIPELINE_DEPTH; d++)
        {
#pragma HLS PIPELINE

            if (tile_in_d + d >= layer_ifms_depth || (model_configs_list_limit != 0 && tile_in_d + d >= model_configs_list_limit))
            {
                break;
            }

            const int main_tile_index = (tile_in_d + d) * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_w + tile_in_w;
            const int bottom_tile_index = main_tile_index + num_of_ifm_tiles_w;
            const int top_tile_index = main_tile_index - num_of_ifm_tiles_w;
            // top and bottom
            for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                if (h >= padding_top_left)
                {
                    break;
                }
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    if (tile_in_h == 0 || tile_in_w * CHANNELS_TILE_WIDTH + w >= layer_ifm_width)
                    {
                        channels_tile[d][h][w + padding_top_left] = current_layer_zero_point;
                    }
                    else
                    {
                        channels_tile[d][h][w + padding_top_left] =
                            channels[top_tile_index][CHANNELS_TILE_HEIGHT - (padding_top_left - h)][w];
                    }
                }
            }

            for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                if (h >= padding_bottom_right)
                {
                    break;
                }
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    if (tile_in_h == num_of_ifm_tiles_h - 1 || tile_in_w * CHANNELS_TILE_WIDTH + w >= layer_ifm_width)
                    {
                        channels_tile[d][h + CHANNELS_TILE_HEIGHT + padding_top_left][w + padding_top_left] = current_layer_zero_point;
                    }
                    else
                    {
                        channels_tile[d][h + CHANNELS_TILE_HEIGHT + padding_top_left][w + padding_top_left] =
                            channels[bottom_tile_index][h][w];
                    }
                }
            }
        }

        for (int d = 0; d < CHANNELS_PIPELINE_DEPTH; d++)
        {
#pragma HLS PIPELINE

            if (tile_in_d + d >= layer_ifms_depth || (model_configs_list_limit != 0 && tile_in_d + d >= model_configs_list_limit))
            {
                break;
            }

            const int main_tile_index = (tile_in_d + d) * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_w + tile_in_w;
            const int right_tile_index = main_tile_index + 1;
            const int left_tile_index = main_tile_index - 1;
            // left and right
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_top_left)
                    {
                        break;
                    }
                    if (tile_in_w == 0 || tile_in_h * CHANNELS_TILE_HEIGHT + h >= layer_ifm_height)
                    {
                        channels_tile[d][h + padding_top_left][w] = current_layer_zero_point;
                    }
                    else
                    {
                        channels_tile[d][h + padding_top_left][w] =
                            channels[left_tile_index][h][CHANNELS_TILE_WIDTH - (padding_top_left - w)];
                    }
                }
            }

            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    if (w >= padding_bottom_right)
                    {
                        break;
                    }
                    if (tile_in_w == num_of_ifm_tiles_w - 1 || tile_in_h * CHANNELS_TILE_HEIGHT + h >= layer_ifm_height)
                    {
                        channels_tile[d][h + padding_top_left][w + padding_top_left + CHANNELS_TILE_WIDTH] = current_layer_zero_point;
                    }
                    else
                    {
                        channels_tile[d][h + padding_top_left][w + padding_top_left + CHANNELS_TILE_WIDTH] =
                            channels[right_tile_index][h][w];
                    }
                }
            }
        }
    }

    for (int d = 0; d < CHANNELS_PIPELINE_DEPTH; d++)
    {
#pragma HLS PIPELINE

        if (tile_in_d + d >= layer_ifms_depth || (model_configs_list_limit != 0 && tile_in_d + d >= model_configs_list_limit))
        {
            break;
        }
        
        // fill body
        const int main_tile_index = (tile_in_d + d) * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_w + tile_in_w;
        for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
            {
#pragma HLS UNROLL
                if (tile_in_h * CHANNELS_TILE_HEIGHT + h >= layer_ifm_height || tile_in_w * CHANNELS_TILE_WIDTH + w >= layer_ifm_width)
                {
                    channels_tile[d][h + padding_top_left][w + padding_top_left] = current_layer_zero_point;
                }
                else
                {
                    channels_tile[d][h + padding_top_left][w + padding_top_left] =
                        channels[main_tile_index][h][w];
                }
            }
        }
    }
}

void copy_fms_tile_corners(fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_TOP_LEFT],
                           fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                           fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
                           fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
                           fms_dt padding_top_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_left_buffer_dst[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_TOP_LEFT],
                           fms_dt padding_top_left_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                           fms_dt padding_top_right_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_left_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                           fms_dt padding_bottom_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                           fms_dt padding_right_buffer_dst[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
                           fms_dt padding_bottom_right_buffer_dst[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
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
            const int d = o_d * CHANNELS_TILE_DEPTH + i_d;
            if (starting_d + d >= ifms_d)
            {
                break;
            }
            // bottom right corner
            for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    padding_bottom_right_buffer_dst[d][h][w] =
                        padding_bottom_right_buffer[d][h][w];
                }
            }
            // other corners
            for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
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
            for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    padding_top_buffer_dst[d][h][w] = padding_top_buffer[d][h][w];
                }
            }
            for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    padding_bottom_buffer_dst[d][h][w] = padding_bottom_buffer[d][h][w];
                }
            }
            // right and left
            for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
                {
#pragma HLS UNROLL
                    padding_right_buffer_dst[d][h][w] = padding_right_buffer[d][h][w];
                }
                for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
                {
#pragma HLS UNROLL
                    padding_left_buffer_dst[d][h][w] = padding_left_buffer[d][h][w];
                }
            }
        }
    }
}

void fill_fms_tile(fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                   fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
                   fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_TOP_LEFT],
                   fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                   fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                   fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
                   fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
                   fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
                   fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
                   fms_dt channels_tile[CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
                   const int tile_in_d,
                   const int tile_in_h,
                   const int tile_in_w,
                   const int num_of_ifm_tiles_h,
                   const int num_of_ifm_tiles_w,
                   const int layer_ifm_height,
                   const int layer_ifm_width,
                   const int padding_top_left,
                   const fms_dt current_layer_zero_point)
{
#pragma HLS INLINE

    const int d = tile_in_d % CHANNELS_PIPELINE_DEPTH;
    // bottom right corner
    for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
        {
#pragma HLS UNROLL
            channels_tile[h + padding_top_left + CHANNELS_TILE_HEIGHT]
                         [w + padding_top_left + CHANNELS_TILE_WIDTH] =
                             padding_bottom_right_buffer[d][h][w];
        }
    }
    // other corners
    for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
        {
#pragma HLS UNROLL
            channels_tile[h][w] = padding_top_left_buffer[d][h][w];
            channels_tile[h][w + padding_top_left + CHANNELS_TILE_WIDTH] =
                padding_top_right_buffer[d][h][w];
            channels_tile[h + padding_top_left + CHANNELS_TILE_HEIGHT][w] =
                padding_bottom_left_buffer[d][h][w];
        }
    }
    // top and bottom
    for (int h = 0; h < MAX_TILE_PADDING_TOP_LEFT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
        {
#pragma HLS UNROLL
            channels_tile[h][w + padding_top_left] = padding_top_buffer[d][h][w];
        }
    }
    for (int h = 0; h < MAX_TILE_PADDING_BOTTOM_RIGHT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
        {
#pragma HLS UNROLL
            channels_tile[h + CHANNELS_TILE_HEIGHT + padding_top_left][w + padding_top_left] =
                padding_bottom_buffer[d][h][w];
        }
    }
    // left and right
    for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < MAX_TILE_PADDING_TOP_LEFT; w++)
        {
#pragma HLS UNROLL
            channels_tile[h + padding_top_left][w] = padding_left_buffer[d][h][w];
        }
    }
    for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < MAX_TILE_PADDING_BOTTOM_RIGHT; w++)
        {
#pragma HLS UNROLL
            channels_tile[h + padding_top_left][w + padding_top_left + CHANNELS_TILE_WIDTH] =
                padding_right_buffer[d][h][w];
        }
    }
    // fill body
    const int num_of_tiles_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w;
    const int main_tile_index = tile_in_d * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_w + tile_in_w;
    for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
        {
#pragma HLS UNROLL
            if (tile_in_h * CHANNELS_TILE_HEIGHT + h >= layer_ifm_height || tile_in_w * CHANNELS_TILE_WIDTH + w >= layer_ifm_width)
            {
                channels_tile[h + padding_top_left][w + padding_top_left] = current_layer_zero_point;
            }
            else
            {
                channels_tile[h + padding_top_left][w + padding_top_left] =
                    channels[main_tile_index][h][w];
            }
        }
    }
}

int get_layer_index_from_offset(const int anchor_layer, const int offset)
{

    int last_layer_index = -1;
    int layer_index;
    int conv_layers_so_far = 0;
    layer_specs tmp;
    for (int i = 0; i < max_conv_layers; i++)
    {
        layer_index = i;
        get_layer_specs_from_index(layer_index, tmp);
        if (layer_index != -1)
        {
            conv_layers_so_far++;
        }
        last_layer_index = i;
        if (conv_layers_so_far == offset)
        {
            break;
        }
    }
    return last_layer_index;
}