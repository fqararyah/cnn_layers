
#include "../headers/conv_utils.h"

void padd_fms_tile_top_left(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                            fms_dt channels_tile[CHANNELS_TILE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                            const int starting_d,
                            const int tile_in_h,
                            const int tile_in_w,
                            const int padding_top_left,
                            const int num_of_ifm_tiles_d,
                            const int num_of_ifm_tiles_h,
                            const int num_of_ifm_tiles_w,
                            fms_dt fms_zero_point)
{
#pragma HLS INLINE

    const int num_of_tiles_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w;
    const int main_tile_index = starting_d * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_h + tile_in_w;
    const int top_tile_index = main_tile_index - num_of_ifm_tiles_w;
    const int left_tile_index = main_tile_index - 1;
    const int top_left_tile_index = top_tile_index - 1;
    const int top_right_tile_index = top_tile_index + 1;
    const int bottom_left_tile_index = num_of_ifm_tiles_w - 1;

    for (int d = 0; d < CHANNELS_TILE_DEPTH; d++)
    {
#pragma HLS PIPELINE
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
                    channels_tile[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
                                 [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] = fms_zero_point;
                }
                else
                {
                    channels_tile[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
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
                    channels_tile[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
                                 [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] = fms_zero_point;
                }
                else
                {
                    channels_tile[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
                                 [w + CHANNELS_TILE_WIDTH + MAX_PADDING_TOP_LEFT] =
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
                    channels_tile[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
                                 [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] = fms_zero_point;
                }
                else
                {
                    channels_tile[d][h + CHANNELS_TILE_HEIGHT + MAX_PADDING_TOP_LEFT]
                                 [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] =
                                     channels[top_left_tile_index][h]
                                             [CHANNELS_TILE_WIDTH - (padding_top_left - w)];
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
                channels_tile[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)][w + MAX_PADDING_TOP_LEFT] =
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
                channels_tile[d][MAX_PADDING_TOP_LEFT + h]
                             [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] =
                                 channels[left_tile_index][h][CHANNELS_TILE_WIDTH - (padding_top_left - w)];
            }
        }
    }
}

void padd_fms_tile_top_right(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                             fms_dt channels_tile[CHANNELS_TILE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                             const int starting_d,
                             const int tile_in_h,
                             const int tile_in_w,
                             const int padding_top_left,
                             const int num_of_ifm_tiles_d,
                             const int num_of_ifm_tiles_h,
                             const int num_of_ifm_tiles_w,
                             fms_dt fms_zero_point)
{
#pragma HLS INLINE

    const int num_of_tiles_hw = num_of_ifm_tiles_h * num_of_ifm_tiles_w;
    const int main_tile_index = starting_d * num_of_tiles_hw + tile_in_h * num_of_ifm_tiles_h + tile_in_w;
    const int bottom_tile_index = main_tile_index - num_of_ifm_tiles_w;
    const int right_tile_index = main_tile_index - 1;
    const int bottom_right_tile_index = bottom_tile_index + 1; 

    for (int d = 0; d < CHANNELS_TILE_DEPTH; d++)
    {
#pragma HLS PIPELINE
        // bottom right corner
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
                if (tile_in_h == num_of_ifm_tiles_h - 1 || tile_in_w == num_of_ifm_tiles_w - 1)
                {
                    channels_tile[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)]
                                 [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] = fms_zero_point;
                }
                else
                {
                    channels_tile[d][h + CHANNELS_TILE_HEIGHT + MAX_PADDING_TOP_LEFT]
                                 [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] =
                                     channels[bottom_right_tile_index][h]
                                             [CHANNELS_TILE_WIDTH - (padding_top_left - w)];
                }
            }
        }
        // bottom
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
                channels_tile[d][h + (MAX_PADDING_TOP_LEFT - padding_top_left)][w + MAX_PADDING_TOP_LEFT] =
                    channels[bottom_tile_index][h][w];
            }
        }
        // right
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
                channels_tile[d][MAX_PADDING_TOP_LEFT + h]
                             [w + (MAX_PADDING_TOP_LEFT - padding_top_left)] =
                                 channels[right_tile_index][h][w];
            }
        }
    }
}

void fill_fms_tile(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                   fms_dt channels_tile[CHANNELS_TILE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                   conv_type layer_type,
                   const int starting_d,
                   const int tile_in_h,
                   const int tile_in_w,
                   const int padding_top_left,
                   const int num_of_ifm_tiles_d,
                   const int num_of_ifm_tiles_h,
                   const int num_of_ifm_tiles_w,
                   fms_dt fms_zero_point)
{
#pragma HLS INLINE off

    for (int d = starting_d; d < CHANNELS_TILE_DEPTH; d++)
    {
#pragma HLS PIPELINE
        for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
            {
#pragma HLS UNROLL
                channels_tile[d][MAX_PADDING_TOP_LEFT + h][MAX_PADDING_TOP_LEFT + w] =
                    channels[d + starting_d][h][w];
            }
        }
    }
}