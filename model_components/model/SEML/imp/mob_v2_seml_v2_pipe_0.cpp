#include "../headers/seml.h"

#if HW == CPU
#include "../../../../tests/test_utils.h"
#endif

using namespace seml_engines;

#if FIBHA_VERSION == 2 && MODEL_ID == MOB_V2 && PIPELINE_LENGTH == 0

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
          const int layer_to_produce_row_counts[max_conv_layers])
{
#pragma HLS INLINE off

    fms_dt switching_buffer[MAX_TMP_FMS_BUFFER_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH];
    // fms_dt tmp_channels2[max_tmp_fms_size];

#pragma HLS ARRAY_PARTITION variable = switching_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = switching_buffer type = complete dim = 3

    //		begin_code_generation
    // #if DEBUGGING
    //     fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/ifms_2.txt",
    //                      channels, layer_2_dw_specs);
    // #endif
    // #if DEBUGGING
    //     verify_fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_2.txt",
    //                             channels, layer_2_dw_specs);
    // #endif

    for (int starting_row = 0; starting_row < 7; starting_row++)
    {
        layer_0_s_3x3(input_image, channels,
                      starting_row * (layer_to_produce_row_counts[0] - (starting_row > 0) * 3) - ((starting_row > 0)),
                      layer_to_produce_row_counts[0]);

        seml_engines::fill_layer_dw_weights_off_chip(off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[2], layer_2_dw_specs.layer_depth);
        seml_engines::fill_fused_scales(off_chip_fused_scales,
                                        seml_fused_scales_buffer,
                                        layers_fused_parameters_offsets[2],
                                        layer_2_dw_specs.layer_num_fils);
        seml_engines::fill_fused_zero_points(off_chip_fused_zero_points,
                                             seml_fused_zero_points_buffer,
                                             layers_fused_parameters_offsets[2],
                                             layer_2_dw_specs.layer_num_fils);
        seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 2, layer_2_dw_specs,
                                  seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                  model_configs_list, layer_to_produce_row_counts[2],
                                  starting_row * (layer_to_produce_row_counts[2] - (starting_row > 0) * 1));

        for (int tile_in_h = 0; tile_in_h < 5; tile_in_h++)
        {
            for (int h = 0; h < dw_tile_h; h++)
            {
                for (int tile_in_w = 0; tile_in_w < 4; tile_in_w++)
                {
                    for (int w = 0; w < dw_tile_w; w++)
                    {
                        printf("%d ", result[tile_in_h * first_conv_layer_specs.layer_num_of_ofm_tiles_w + tile_in_w][h][w]);
                    }
                }
                printf("\n");
            }
        }
        printf("******************\n");
#if DEBUGGING
        dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_2.txt",
                          result, layer_2_dw_specs);
#endif
        seml_engines::fill_fused_scales(off_chip_fused_scales,
                                        seml_fused_scales_buffer,
                                        layers_fused_parameters_offsets[3],
                                        layer_3_pw_specs.layer_num_fils);
        seml_engines::fill_fused_zero_points(off_chip_fused_zero_points,
                                             seml_fused_zero_points_buffer,
                                             layers_fused_parameters_offsets[3],
                                             layer_3_pw_specs.layer_num_fils);
        pw_conv(off_chip_weights, result, channels, tmp_channels, 3, layer_3_pw_specs,
                seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                model_configs_list, layer_to_produce_row_counts[3]);

#if DEBUGGING
        dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_3.txt",
                          channels, layer_3_pw_specs);
#endif
        seml_engines::fill_fused_scales(off_chip_fused_scales,
                                        seml_fused_scales_buffer,
                                        layers_fused_parameters_offsets[4],
                                        layer_4_pw_specs.layer_num_fils);
        seml_engines::fill_fused_zero_points(off_chip_fused_zero_points,
                                             seml_fused_zero_points_buffer,
                                             layers_fused_parameters_offsets[4],
                                             layer_4_pw_specs.layer_num_fils);
        pw_conv(off_chip_weights, channels, result, tmp_channels, 4, layer_4_pw_specs,
                seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                model_configs_list, layer_to_produce_row_counts[4]);

#if DEBUGGING
        dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_4.txt",
                          result, layer_4_pw_specs);
#endif
        seml_engines::fill_layer_dw_weights_off_chip(off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[6], layer_6_dw_specs.layer_depth);
        seml_engines::fill_fused_scales(off_chip_fused_scales,
                                        seml_fused_scales_buffer,
                                        layers_fused_parameters_offsets[6],
                                        layer_6_dw_specs.layer_num_fils);
        seml_engines::fill_fused_zero_points(off_chip_fused_zero_points,
                                             seml_fused_zero_points_buffer,
                                             layers_fused_parameters_offsets[6],
                                             layer_6_dw_specs.layer_num_fils);
        seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 6, layer_6_dw_specs,
                                  seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                                  model_configs_list, layer_to_produce_row_counts[6]);
#if DEBUGGING
        dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_6.txt",
                          channels, layer_6_dw_specs);
#endif
        seml_engines::fill_fused_scales(off_chip_fused_scales,
                                        seml_fused_scales_buffer,
                                        layers_fused_parameters_offsets[7],
                                        layer_7_pw_specs.layer_num_fils);
        seml_engines::fill_fused_zero_points(off_chip_fused_zero_points,
                                             seml_fused_zero_points_buffer,
                                             layers_fused_parameters_offsets[7],
                                             layer_7_pw_specs.layer_num_fils);
        pw_conv(off_chip_weights, channels, result, tmp_channels, 7, layer_7_pw_specs,
                seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
                model_configs_list, layer_to_produce_row_counts[7]);

        const int num_produced_tiles_h = layer_to_produce_row_counts[7] / pw_tile_h;
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
                            switching_buffer[tile_index_in_src + offset_h_tiles][h][w] = result[tile_index_in_src][h][w];
                        }
                    }
                }
            }
        }
    }
#if DEBUGGING
    dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
                      switching_buffer, layer_7_pw_specs);
#endif
    seml_engines::fill_fused_scales(off_chip_fused_scales,
                                    seml_fused_scales_buffer,
                                    layers_fused_parameters_offsets[8],
                                    layer_8_pw_specs.layer_num_fils);
    seml_engines::fill_fused_zero_points(off_chip_fused_zero_points,
                                         seml_fused_zero_points_buffer,
                                         layers_fused_parameters_offsets[8],
                                         layer_8_pw_specs.layer_num_fils);
    pw_conv(off_chip_weights, result, channels, tmp_channels, 8, layer_8_pw_specs,
            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
            model_configs_list, layer_to_produce_row_counts[8]);

// 	//end_code_generation
#if MODEL_ID == RESNET50
    avgpool(channels, fc_input, layer_73_avgpool_specs);
#elif MODEL_ID == MOB_V2
    avgpool(result, fc_input, layer_67_avgpool_specs);
#endif
    // fc_layer(fc_weights, fc_input, fc_output);
}

#endif