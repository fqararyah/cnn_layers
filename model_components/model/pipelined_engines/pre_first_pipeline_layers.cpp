#include "pipeline_main.h"

void fill_input_image_groups_buffer(
    fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
    fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * INPUT_IMAGE_ROWS_FILLED_EACH_TIME],
    int starting_h,
    const int elements_to_fill_from_an_ifm)
{ // chain_0_1_layer_0_s_in_rows_at_once * input_image_num_fms_groups_in_width
#pragma HLS INLINE off

    const int start_filling_offset = starting_h * input_image_num_fms_groups_in_width;
    int elements_avaiable_in_input_image;

    elements_avaiable_in_input_image = (input_image_height - starting_h) * input_image_num_fms_groups_in_width;

    for (int d = 0; d < input_image_depth; d++)
    {
#pragma HLS PIPELINE off
        const int d_offst = start_filling_offset + d * input_image_num_fms_groups_in_a_channel;
        for (int i = 0; i < elements_to_fill_from_an_ifm; i++)
        {
#pragma HLS PIPELINE off
            if (i < elements_avaiable_in_input_image)
            {
                fms_groups_buffer[d][i] = channels[d_offst + i];
            }
        }
    }
}

void input_image_fill_row_from_groups_buffer(
    fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * INPUT_IMAGE_ROWS_FILLED_EACH_TIME],
    fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
    int row, const int channels_buffer_start_filling_h)
{
#pragma HLS INLINE

    const int start_filling_offset = row * input_image_num_fms_groups_in_width;
    for (int o_w = 0; o_w < input_image_num_fms_groups_in_width; o_w++)
    {
#pragma HLS PIPELINE off
        const int o_w_offset = o_w * input_image_group_items;
        for (int d = 0; d < input_image_depth; d++)
        {
#pragma HLS UNROLL
            fms_grp_dt chunck = fms_groups_buffer[d][start_filling_offset + o_w];
            for (int w = 0; w < input_image_group_items; w++)
            {
#pragma HLS PIPELINE
                if (o_w_offset + w < input_image_width)
                {
#if HW == _FPGA
                    channels_buffer_0[d][channels_buffer_start_filling_h + row][o_w_offset + w] = (fms_dt)chunck(
                        w * fms_dt_width + fms_dt_offset, w * fms_dt_width);
#endif
                }
            }
        }
    }
}

void input_image_shift_channels_buffer_rows(
    fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
    const int rows_to_shift)
{
#pragma HLS INLINE

    for (int w = 0; w < input_image_width; w++)
    {
#pragma HLS PIPELINE
        for (int d = 0; d < input_image_depth; d++)
        {
#pragma HLS UNROLL
            for (int h = 0; h < rows_to_shift; h++)
            {
#pragma HLS UNROLL
                channels_buffer_0[d][h][w] = channels_buffer_0[d][h + INPUT_IMAGE_ROWS_FILLED_EACH_TIME][w];
            }
        }
    }
}

void input_image_padd_bottom_channels_buffer_rows(
    fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
    const fms_dt zero_point)
{
#pragma HLS INLINE

    for (int w = 0; w < input_image_width; w++)
    {
#pragma HLS PIPELINE
        for (int d = 0; d < input_image_depth; d++)
        {
#pragma HLS UNROLL
            for (int h = PRE_FIRST_PIPELINE_INPUT_HEIGHT - first_conv_layer_specs.padding_bottom;
                 h < PRE_FIRST_PIPELINE_INPUT_HEIGHT; h++)
            {
#pragma HLS UNROLL
                channels_buffer_0[d][h][w] = zero_point;
            }
        }
    }
}

void input_image_fill_channels_buffer_from_groups_buffer(
    fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * INPUT_IMAGE_ROWS_FILLED_EACH_TIME],
    fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
    int starting_h, bool shift, const fms_dt zero_point)
{
#pragma HLS INLINE off

    const int rows_to_shift = PRE_FIRST_PIPELINE_INPUT_HEIGHT - INPUT_IMAGE_ROWS_FILLED_EACH_TIME;
    if (shift)
    {
        input_image_shift_channels_buffer_rows(channels_buffer_0, rows_to_shift);
    }
    const int channels_buffer_start_filling_h =
        starting_h == 0 ? first_conv_layer_specs.padding_top : rows_to_shift;
    for (int h = 0; h < INPUT_IMAGE_ROWS_FILLED_EACH_TIME; h++)
    {
        if (starting_h + h < input_image_height)
        {
            input_image_fill_row_from_groups_buffer(fms_groups_buffer,
                                                    channels_buffer_0, h, channels_buffer_start_filling_h);
        }
        else
        {
            input_image_padd_bottom_channels_buffer_rows(channels_buffer_0,
                                                         zero_point);
        }
    }
}

void fill_first_cols_of_first_layer_input(
    fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
    fms_dt first_layer_input_buffer[FIRST_CONV_LAYER_BUFFER_SIZE],
    int starting_h)
{

    const int first_layer_input_buffer_hw = first_conv_layer_filter_dim * first_conv_layer_filter_dim;
fill_first_cols_of_first_layer_input:
    for (int d = 0;
         d < input_image_depth; d++)
    {
        for (int h = 0; h < PRE_FIRST_PIPELINE_INPUT_HEIGHT; h++)
        {
            for (int w = 0;
                 w < first_conv_layer_filter_dim - first_conv_layer_strides;
                 w++)
            {
                first_layer_input_buffer[d * first_layer_input_buffer_hw + h * first_conv_layer_filter_dim +
                                         w] =
                    channels_buffer_0[d][starting_h + h][w];
            }
        }
    }
}

void fill_first_layer_input_new_cols(
    fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
    fms_dt first_layer_input_new_cols[FIRST_CONV_LAYER_NEW_COLS_BUFFER_SIZE],
    const int starting_w, int starting_h, fms_dt zero_point)
{

    const int buffer_hw = first_conv_layer_filter_dim * first_conv_layer_strides;
    for (int d = 0; d < input_image_depth; d++)
    {
        for (int h = 0; h < first_conv_layer_filter_dim; h++)
        {
            for (int w = 0; w < first_conv_layer_strides; w++)
            {
                if (starting_w + w < input_image_width + first_conv_layer_padding_left)
                {
                    first_layer_input_new_cols[d * buffer_hw + h * first_conv_layer_strides + w] =
                        channels_buffer_0[d][h + starting_h][starting_w + w];
                }
                else
                {
                    first_layer_input_new_cols[d * buffer_hw + h * first_conv_layer_strides + w] =
                        zero_point;
                }
            }
        }
    }
}

void shift_and_fill_first_layer_input(
    fms_dt first_layer_input_new_cols[FIRST_CONV_LAYER_NEW_COLS_BUFFER_SIZE],
    fms_dt first_layer_input[FIRST_CONV_LAYER_BUFFER_SIZE])
{
#pragma HLS INLINE

    const int to_be_shifted = first_conv_layer_filter_dim - first_conv_layer_strides;
    const int first_layer_input_buffer_hw = first_conv_layer_filter_dim * first_conv_layer_filter_dim;

fill_bottleneck_0_input:
    for (int d = 0; d < input_image_depth; d++)
    {
#pragma HLS UNROLL
        for (int h = 0; h < first_conv_layer_filter_dim; h++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < to_be_shifted; w++)
            {
#pragma HLS UNROLL
                const int to_fill_in_index = d * first_layer_input_buffer_hw + h * first_conv_layer_filter_dim + w;
                first_layer_input[to_fill_in_index] =
                    first_layer_input[to_fill_in_index + first_conv_layer_strides];
            }
            for (int w = 0; w < first_conv_layer_strides; w++)
            {
#pragma HLS UNROLL
                first_layer_input[d * first_layer_input_buffer_hw + h * first_conv_layer_filter_dim + w + to_be_shifted] =
                    first_layer_input_new_cols[d * first_layer_input_buffer_hw + h * first_conv_layer_filter_dim + w];
            }
        }
    }
}

void input_image_fill_channels_buffer_cpu(
    fms_dt channels[input_image_depth * input_image_hw],
    fms_dt channels_buffer_0[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
    int starting_h, bool shift, const fms_dt zero_point)
{
#pragma HLS INLINE off

    const int rows_to_shift = FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME;
    if (shift)
    {
        input_image_shift_channels_buffer_rows(channels_buffer_0, rows_to_shift);
    }
    const int channels_buffer_start_filling_h =
        starting_h == 0 ? first_conv_layer_specs.padding_top : rows_to_shift;
    for (int h = 0; h < INPUT_IMAGE_ROWS_FILLED_EACH_TIME; h++)
    {
        if (starting_h + h < input_image_height)
        {
            for (int d = 0; d < input_image_depth; d++)
            {
                for (int w = 0; w < input_image_width; w++)
                {
                    channels_buffer_0[d][h + channels_buffer_start_filling_h][w] =
                        channels[d * input_image_hw + (h + starting_h) * input_image_width + w];
                }
            }
        }
        else
        {
            input_image_padd_bottom_channels_buffer_rows(channels_buffer_0,
                                                         zero_point);
        }
    }
}

void first_conv_and_dw_layers_pipeline(fms_dt first_layer_input[FIRST_CONV_LAYER_BUFFER_SIZE],
                                       weights_dt dw_layer_weights[layer_2_dw_num_fils][layer_2_dw_filter_dim * layer_2_dw_filter_dim],
                                       fms_dt conv_dw_communication_buffer_inter[first_conv_layer_num_fils][layer_2_dw_filter_dim]
                                                                                [layer_2_dw_ifm_width],
                                       fms_dt conv_dw_communication_buffer_intra[first_conv_layer_num_fils]
                                                                                [layer_2_dw_filter_dim][layer_2_dw_filter_dim],
                                       fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH]
                                                                              [PRE_FIRST_PIPELINE_OUTPUT_HEIGHT]
                                                                              [PRE_FIRST_PIPELINE_OUTPUT_WIDTH],
                                       const int starting_h, const int starting_w,
                                       const int writing_row,
                                       fms_quantization_scheme first_conv_layer_quantization_params[first_conv_layer_num_fils],
                                       fms_quantization_scheme first_dw_layer_quantization_params[layer_2_dw_num_fils])
{
#pragma HLS INLINE off

    const int dw_starting_h = starting_h - (layer_2_dw_filter_dim - 1);
    const int dw_starting_w = starting_w - (layer_2_dw_filter_dim - 1);

    for (int f = 0; f < first_conv_layer_num_fils; f++)
    {
#pragma HLS PIPELINE
        conv_dw_communication_buffer_inter[f][writing_row][starting_w] =
            first_layer_conv_kernel(first_layer_input, first_layer_weights, f, starting_h, starting_w,
                                    first_conv_layer_quantization_params[f]);
        if (dw_starting_h >= 0 && dw_starting_w >= 0)
        {
            pre_first_pipeline_layers_output[f][dw_starting_h][dw_starting_w] =
                first_dw_layer_kernel(conv_dw_communication_buffer_intra, dw_layer_weights, layer_2_dw_filter_dim, f,
                                      first_dw_layer_quantization_params[f], layer_2_dw_specs.layer_activation);
        }
    }
}

void pre_first_pipeline_layers_mob_v2(fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
                                      fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH]
                                                                             [PRE_FIRST_PIPELINE_OUTPUT_HEIGHT]
                                                                             [PRE_FIRST_PIPELINE_OUTPUT_WIDTH],
                                      weights_dt dw_layer_weights[layer_2_dw_num_fils][layer_2_dw_filter_dim * layer_2_dw_filter_dim],
                                      fms_quantization_scheme first_layer_quantization_params[first_conv_layer_num_fils],
                                      fms_quantization_scheme first_dw_layer_quantization_params[layer_2_dw_num_fils],
                                      fms_dt conv_dw_communication_buffer_inter[first_conv_layer_num_fils][layer_2_dw_filter_dim]
                                                                               [layer_2_dw_ifm_width],
                                      const int starting_h)
{

    const int num_of_ifm_groups_read_each_time =
        input_image_num_fms_groups_in_width * INPUT_IMAGE_ROWS_FILLED_EACH_TIME;

    fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * INPUT_IMAGE_ROWS_FILLED_EACH_TIME];
    fms_dt first_layers_input[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width];

#if HW == _FPGA
    fill_input_image_groups_buffer(channels, fms_groups_buffer,
                                   0,
                                   input_image_num_fms_groups_in_width); // to do
    input_image_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
                                                        first_layers_input, 0, false,
                                                        first_conv_layer_specs.layer_ifms_zero_point);
    fill_input_image_groups_buffer(channels, fms_groups_buffer,
                                   FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME,
                                   num_of_ifm_groups_read_each_time); // to do
    input_image_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
                                                        first_layer_input, FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME, false,
                                                        first_conv_layer_specs.layer_ifms_zero_point);
#elif HW == CPU
    input_image_fill_channels_buffer_cpu(channels, first_layers_input, 0, false, first_conv_layer_specs.layer_ifms_zero_point);
    input_image_fill_channels_buffer_cpu(channels,
                                         first_layers_input, FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME, false,
                                         first_conv_layer_specs.layer_ifms_zero_point);
#endif

    fms_dt first_layer_input_buffer[FIRST_CONV_LAYER_BUFFER_SIZE];
    fms_dt first_layer_input_buffer_new_cols[FIRST_CONV_LAYER_BUFFER_SIZE];

    fms_dt conv_dw_communication_buffer_intra[first_conv_layer_num_fils]
                                             [layer_2_dw_filter_dim][layer_2_dw_filter_dim];

    int writing_row = starting_h % layer_2_dw_filter_dim;

pre_first_pipeline_layers_mob_v2:
    for (int h = 0; h < PRE_FIRST_PIPELINE_OUTPUT_HEIGHT / first_conv_layer_strides; h++)
    {
        fill_first_cols_of_first_layer_input(first_layers_input, first_layer_input_buffer, h * first_conv_layer_strides);
        shift_and_fill_first_layer_input(first_layer_input_buffer_new_cols, first_layer_input_buffer);
        for (int w = 0; w < input_image_dt_width / first_conv_layer_strides; w++)
        {
            first_conv_and_dw_layers_pipeline(first_layer_input_buffer, dw_layer_weights,
                                              conv_dw_communication_buffer_inter,
                                              conv_dw_communication_buffer_intra,
                                              pre_first_pipeline_layers_output,
                                              h, w, writing_row,
                                              first_layer_quantization_params,
                                              first_dw_layer_quantization_params);
            fill_first_layer_input_new_cols(first_layers_input, first_layer_input_buffer_new_cols, w * first_conv_layer_strides,
                                            h * first_conv_layer_strides, first_conv_layer_specs.layer_ifms_zero_point);
            shift_and_fill_first_layer_input(first_layer_input_buffer_new_cols, first_layer_input_buffer);
        }
        writing_row++;
        if (writing_row == 3)
        {
            writing_row = 0;
        }
    }
}