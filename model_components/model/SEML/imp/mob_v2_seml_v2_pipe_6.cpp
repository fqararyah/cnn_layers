#include "../headers/seml.h"

#if HW == CPU
#include "../../../../tests/test_utils.h"
#endif

using namespace seml_engines;

#if FIBHA_VERSION == 2 && MODEL_ID == MOB_V2 && PIPELINE_LENGTH == 6

void run_layers_in_range(weights_grp_dt off_chip_weights[all_off_chip_pw_s_weights],
                         weights_dt off_chip_dw_weights[all_dw_off_chip_weights],
                         fused_scales_dt off_chip_fused_scales[all_off_chip_fused_scales_zps],
                         biases_dt off_chip_fused_zero_points[all_off_chip_fused_scales_zps],
                         fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                         fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                         fms_dt tmp_channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                         int model_configs_list[2 * max_conv_layers],
                         const soft_pipe_specs_struct soft_pipe_specs[max_conv_layers],
                         const int starting_row,
                         const int starting_layer, const int end_layer, int &even_odd)
{

#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable = seml_dw_weights_3x3 type = cyclic factor = 2 dim = 1

    int layer_index;
    layer_specs current_layer_specs;
    for (int i = starting_layer; i <= end_layer; i++)
    {
#pragma HLS PIPELINE off

        layer_index = i;
        get_layer_specs_from_index(layer_index, current_layer_specs);
        if (layer_index != -1)
        {
            //printf(">>>>> %d\n", layer_index);
            if (current_layer_specs.conv_layer_type == DW_CONV)
            {
                seml_engines::fill_layer_dw_weights_off_chip(off_chip_dw_weights, seml_dw_weights_3x3,
                                                             dw_layers_weights_offsets[layer_index],
                                                             current_layer_specs.layer_depth);
            }
            seml_engines::fill_fused_scales(off_chip_fused_scales,
                                            seml_fused_scales_buffer,
                                            layers_fused_parameters_offsets[layer_index],
                                            current_layer_specs.layer_num_fils);
            seml_engines::fill_fused_zero_points(off_chip_fused_zero_points,
                                                 seml_fused_zero_points_buffer,
                                                 layers_fused_parameters_offsets[layer_index],
                                                 current_layer_specs.layer_num_fils);
            if (even_odd == 1)
            {
                if (current_layer_specs.conv_layer_type == PW_CONV)
                {
                    pw_conv(off_chip_weights, channels, result, tmp_channels, layer_index, current_layer_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list,
                            SOFT_PIPELINE ? soft_pipe_specs[layer_index].to_produce_row_count : current_layer_specs.layer_ofm_height);
                }
                else if (current_layer_specs.conv_layer_type == DW_CONV)
                {
                    seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, layer_index, current_layer_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                              SOFT_PIPELINE ? soft_pipe_specs[layer_index].to_produce_row_count : current_layer_specs.layer_ofm_height,
                                              starting_row *
                                                      (soft_pipe_specs[layer_index].to_produce_row_count -
                                                       (starting_row > 0) * soft_pipe_specs[layer_index].redundant_rows) -
                                                  (starting_row > 0) * soft_pipe_specs[layer_index].unused_first_time);
                }
            }
            else
            {
                if (current_layer_specs.conv_layer_type == PW_CONV)
                {
                    pw_conv(off_chip_weights, result, channels, tmp_channels, layer_index, current_layer_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list,
                            SOFT_PIPELINE ? soft_pipe_specs[layer_index].to_produce_row_count : current_layer_specs.layer_ofm_height);
                }
                else if (current_layer_specs.conv_layer_type == DW_CONV)
                {
                    seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, layer_index, current_layer_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                              SOFT_PIPELINE ? soft_pipe_specs[layer_index].to_produce_row_count : current_layer_specs.layer_ofm_height,
                                              starting_row *
                                                      (soft_pipe_specs[layer_index].to_produce_row_count -
                                                       (starting_row > 0) * soft_pipe_specs[layer_index].redundant_rows) -
                                                  (starting_row > 0) * soft_pipe_specs[layer_index].unused_first_time);
                }
            }
            even_odd = 1 - even_odd;
        }
    }
}

void seml(fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
          weights_grp_dt off_chip_weights[all_off_chip_pw_s_weights],
          weights_dt off_chip_dw_weights[all_dw_off_chip_weights],
          fused_scales_dt off_chip_fused_scales[all_off_chip_fused_scales_zps],
          biases_dt off_chip_fused_zero_points[all_off_chip_fused_scales_zps],
          fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
          fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
          fms_dt tmp_channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
          fms_dt fc_input[fc_layer_input_size],
          int model_configs_list[2 * max_conv_layers],
          const soft_pipe_specs_struct soft_pipe_specs[max_conv_layers],
          const int soft_pipeline_len)
{
#pragma HLS INLINE off

#if SOFT_PIPELINE
    fms_dt switching_buffer[MAX_TMP_FMS_BUFFER_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH];
    // fms_dt tmp_channels2[max_tmp_fms_size];

#pragma HLS ARRAY_PARTITION variable = switching_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = switching_buffer type = complete dim = 3
#endif

    // #if DEBUGGING
    //     fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/ifms_2.txt",
    //                      channels, layer_2_dw_specs);
    // #endif
    // #if DEBUGGING
    //     verify_fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_2.txt",
    //                             channels, layer_2_dw_specs);
    // #endif

    int even_odd = 1;
    int last_layer_in_hard_pipe_index = PIPELINE_LENGTH >= first_conv_layer_specs.layer_index
                                            ? get_layer_index_from_offset(0, PIPELINE_LENGTH)
                                            : first_conv_layer_specs.layer_index;

    int last_layer_in_soft_pipe_index = SOFT_PIPELINE && soft_pipeline_len > 0
                                            ? get_layer_index_from_offset(0, soft_pipeline_len)
                                            : first_conv_layer_specs.layer_index;

    int last_layer_in_soft_and_hard_pipelines = last_layer_in_soft_pipe_index > last_layer_in_hard_pipe_index
                                                    ? last_layer_in_soft_pipe_index
                                                    : last_layer_in_hard_pipe_index;

    layer_specs last_soft_pipe_layer_specs;
    get_layer_specs_from_index(last_layer_in_soft_pipe_index, last_soft_pipe_layer_specs);

    int rows_to_produce_by_soft_pipe = soft_pipe_specs[last_layer_in_soft_pipe_index].to_produce_row_count;
    int seml_num_iters = SOFT_PIPELINE
                             ? (last_soft_pipe_layer_specs.layer_ofm_height + rows_to_produce_by_soft_pipe - 1) /
                                   rows_to_produce_by_soft_pipe
                             : 1;

    // printf(">>>>>>>>>> %d\n", seml_num_iters);

    for (int starting_row = 0; starting_row < seml_num_iters; starting_row++)
    {
#pragma HLS PIPELINE off
        if (last_layer_in_hard_pipe_index <= first_conv_layer_specs.layer_index)
        {
            const int first_layer_index = first_conv_layer_specs.layer_index;
            layer_0_s_3x3(input_image, channels,
                          starting_row *
                                  (soft_pipe_specs[first_layer_index].to_produce_row_count -
                                   (starting_row > 0) * soft_pipe_specs[first_layer_index].redundant_rows) -
                              ((starting_row > 0) * soft_pipe_specs[first_layer_index].unused_first_time),
                          SOFT_PIPELINE ? soft_pipe_specs[first_layer_index].to_produce_row_count : first_conv_layer_specs.layer_ofm_height);
        }

        // if (starting_row == 0)
        //     for (int tile_in_h = 0; tile_in_h < 5; tile_in_h++)
        //     {
        //         for (int h = 0; h < dw_tile_h; h++)
        //         {
        //             for (int tile_in_w = 0; tile_in_w < 4; tile_in_w++)
        //             {
        //                 for (int w = 0; w < dw_tile_w; w++)
        //                 {
        //                     printf("%d ", channels[tile_in_h * first_conv_layer_specs.layer_num_of_ofm_tiles_w + tile_in_w][h][w]);
        //                 }
        //             }
        //             printf("\n");
        //         }
        //     }
        // printf("******************\n");

        // printf("%d\n", last_layer_in_pipe_index);
#if SOFT_PIPELINE
        if (soft_pipeline_len > 0)
        {
            run_layers_in_range(off_chip_weights,
                                off_chip_dw_weights,
                                off_chip_fused_scales,
                                off_chip_fused_zero_points,
                                channels,
                                result,
                                tmp_channels,
                                model_configs_list,
                                soft_pipe_specs,
                                starting_row,
                                last_layer_in_hard_pipe_index + 1, last_layer_in_soft_pipe_index, even_odd);

            const int num_produced_tiles_h = soft_pipe_specs[7].to_produce_row_count / pw_tile_h;
            const int num_of_ofm_tiles_h = layer_7_pw_specs.layer_num_of_ofm_tiles_h;
            const int num_of_ofm_tiles_w = layer_7_pw_specs.layer_num_of_ofm_tiles_w;
            const int num_of_tiles_hw = num_of_ofm_tiles_h * num_of_ofm_tiles_w;

            const int offset_h_tiles = starting_row * num_produced_tiles_h * num_of_ofm_tiles_w;
            for (int d = 0; d < layer_7_pw_specs.layer_num_fils; d++)
            {
                for (int tile_in_h = 0; tile_in_h < num_produced_tiles_h; tile_in_h++)
                {
                    for (int tile_in_w = 0; tile_in_w < num_of_ofm_tiles_w; tile_in_w++)
                    {
                        const int tile_index_in_src = d * num_of_tiles_hw + tile_in_h * num_of_ofm_tiles_w + tile_in_w;
                        for (int h = 0; h < pw_tile_h; h++)
                        {
                            for (int w = 0; w < pw_tile_w; w++)
                            {
                                if (even_odd)
                                {
                                    switching_buffer[tile_index_in_src + offset_h_tiles][h][w] = channels[tile_index_in_src][h][w];
                                }
                                else
                                {
                                    switching_buffer[tile_index_in_src + offset_h_tiles][h][w] = result[tile_index_in_src][h][w];
                                }
                            }
                        }
                    }
                }
            }
        }
#endif
    }

#if DEBUGGING
    if (SOFT_PIPELINE && soft_pipeline_len > 0)
    {
        dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
                          switching_buffer, layer_7_pw_specs);
    }
#endif

    /*
    The following call of run_layers_in_range could replace the generated code below it. This
    makes the code much more flexible and with no need to change or rebuild as the layers range change
    however, the penalty of using this flexibility is 73% increase in the running time which is huge.
    */
    // run_layers_in_range(off_chip_weights,
    //                     off_chip_dw_weights,
    //                     off_chip_fused_scales,
    //                     off_chip_fused_zero_points,
    //                     channels,
    //                     result,
    //                     tmp_channels,
    //                     model_configs_list,
    //                     soft_pipe_specs,
    //                     0,
    //                     last_layer_in_soft_and_hard_pipelines + 1, LAYER_LIMIT, even_odd);

    
    //		begin_code_generation
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[8],
                                     layer_8_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[8], 
                                layer_8_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 8, layer_8_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_8_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[9], layer_9_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[9],
                                     layer_9_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[9], 
                                layer_9_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 9, layer_9_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_9_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[10],
                                     layer_10_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[10], 
                                layer_10_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 10, layer_10_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_10_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[12],
                                     layer_12_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[12], 
                                layer_12_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 12, layer_12_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_12_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[14], layer_14_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[14],
                                     layer_14_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[14], 
                                layer_14_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 14, layer_14_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_14_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[15],
                                     layer_15_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[15], 
                                layer_15_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 15, layer_15_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_15_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[16],
                                     layer_16_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[16], 
                                layer_16_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 16, layer_16_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_16_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[17], layer_17_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[17],
                                     layer_17_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[17], 
                                layer_17_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 17, layer_17_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_17_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[18],
                                     layer_18_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[18], 
                                layer_18_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 18, layer_18_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_18_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[20],
                                     layer_20_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[20], 
                                layer_20_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 20, layer_20_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_20_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[21], layer_21_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[21],
                                     layer_21_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[21], 
                                layer_21_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 21, layer_21_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_21_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[22],
                                     layer_22_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[22], 
                                layer_22_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 22, layer_22_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_22_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[24],
                                     layer_24_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[24], 
                                layer_24_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 24, layer_24_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_24_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[26], layer_26_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[26],
                                     layer_26_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[26], 
                                layer_26_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 26, layer_26_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_26_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[27],
                                     layer_27_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[27], 
                                layer_27_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 27, layer_27_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_27_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[28],
                                     layer_28_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[28], 
                                layer_28_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 28, layer_28_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_28_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[29], layer_29_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[29],
                                     layer_29_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[29], 
                                layer_29_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 29, layer_29_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_29_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[30],
                                     layer_30_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[30], 
                                layer_30_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 30, layer_30_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_30_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[32],
                                     layer_32_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[32], 
                                layer_32_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 32, layer_32_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_32_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[33], layer_33_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[33],
                                     layer_33_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[33], 
                                layer_33_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 33, layer_33_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_33_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[34],
                                     layer_34_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[34], 
                                layer_34_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 34, layer_34_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_34_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[36],
                                     layer_36_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[36], 
                                layer_36_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 36, layer_36_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_36_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[37], layer_37_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[37],
                                     layer_37_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[37], 
                                layer_37_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 37, layer_37_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_37_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[38],
                                     layer_38_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[38], 
                                layer_38_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 38, layer_38_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_38_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[40],
                                     layer_40_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[40], 
                                layer_40_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 40, layer_40_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_40_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[41], layer_41_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[41],
                                     layer_41_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[41], 
                                layer_41_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 41, layer_41_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_41_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[42],
                                     layer_42_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[42], 
                                layer_42_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 42, layer_42_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_42_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[43],
                                     layer_43_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[43], 
                                layer_43_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 43, layer_43_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_43_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[44], layer_44_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[44],
                                     layer_44_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[44], 
                                layer_44_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 44, layer_44_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_44_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[45],
                                     layer_45_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[45], 
                                layer_45_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 45, layer_45_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_45_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[47],
                                     layer_47_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[47], 
                                layer_47_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 47, layer_47_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_47_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[48], layer_48_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[48],
                                     layer_48_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[48], 
                                layer_48_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 48, layer_48_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_48_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[49],
                                     layer_49_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[49], 
                                layer_49_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 49, layer_49_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_49_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[51],
                                     layer_51_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[51], 
                                layer_51_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 51, layer_51_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_51_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[53], layer_53_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[53],
                                     layer_53_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[53], 
                                layer_53_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 53, layer_53_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_53_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[54],
                                     layer_54_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[54], 
                                layer_54_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 54, layer_54_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_54_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[55],
                                     layer_55_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[55], 
                                layer_55_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 55, layer_55_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_55_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[56], layer_56_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[56],
                                     layer_56_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[56], 
                                layer_56_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 56, layer_56_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_56_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[57],
                                     layer_57_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[57], 
                                layer_57_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 57, layer_57_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_57_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[59],
                                     layer_59_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[59], 
                                layer_59_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 59, layer_59_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_59_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[60], layer_60_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[60],
                                     layer_60_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[60], 
                                layer_60_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 60, layer_60_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_60_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[61],
                                     layer_61_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[61], 
                                layer_61_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 61, layer_61_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_61_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[63],
                                     layer_63_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[63], 
                                layer_63_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 63, layer_63_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_63_pw_specs.layer_ofm_height);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[64], layer_64_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[64],
                                     layer_64_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[64], 
                                layer_64_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 64, layer_64_dw_specs,
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                              model_configs_list,
                                            layer_64_dw_specs.layer_ofm_height, 0);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[65],
                                     layer_65_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[65], 
                                layer_65_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels, result, tmp_channels, 65, layer_65_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_65_pw_specs.layer_ofm_height);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[66],
                                     layer_66_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[66], 
                                layer_66_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result, channels, tmp_channels, 66, layer_66_pw_specs,
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                            model_configs_list, layer_66_pw_specs.layer_ofm_height);

    // 	end_code_generation

#if DEBUGGING
    if (even_odd)
    {
        dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_66.txt",
                          channels, layer_66_pw_specs);
    }
    else
    {
        dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_66.txt",
                          result, layer_66_pw_specs);
    }
#endif

#if MODEL_ID == RESNET50
    avgpool(channels, fc_input, layer_73_avgpool_specs);
#elif MODEL_ID == MOB_V2
    if (even_odd)
    {
        avgpool(channels, fc_input, layer_67_avgpool_specs);
    }
    else
    {
        avgpool(result, fc_input, layer_67_avgpool_specs);
    }
#endif
    // fc_layer(fc_weights, fc_input, fc_output);
}

#endif