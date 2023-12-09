#if DEBUGGING
 fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v1/fms/ifms_15.txt",
 channels, layer_15_dw_specs);
#endif
#if DEBUGGING
 verify_fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_15.txt",
 channels, layer_15_dw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 15,layer_15_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_15.txt",
 result, layer_15_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 16, layer_16_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_16.txt",
 channels, layer_16_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 17,layer_17_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_17.txt",
 result, layer_17_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 18, layer_18_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 19,layer_19_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 20, layer_20_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_20.txt",
 channels, layer_20_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 21,layer_21_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_21.txt",
 result, layer_21_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 22, layer_22_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 23,layer_23_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 24, layer_24_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_24.txt",
 channels, layer_24_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 25,layer_25_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 26, layer_26_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_26.txt",
 channels, layer_26_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 28,layer_28_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 29, layer_29_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 30,layer_30_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 31, layer_31_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 33, layer_33_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);#if DEBUGGING
 fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v1/fms/ifms_15.txt",
 channels, layer_15_dw_specs);
#endif
#if DEBUGGING
 verify_fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_15.txt",
 channels, layer_15_dw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 15,layer_15_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_15.txt",
 result, layer_15_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 16, layer_16_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_16.txt",
 channels, layer_16_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 17,layer_17_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_17.txt",
 result, layer_17_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 18, layer_18_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 19,layer_19_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 20, layer_20_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_20.txt",
 channels, layer_20_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 21,layer_21_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_21.txt",
 result, layer_21_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 22, layer_22_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 23,layer_23_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 24, layer_24_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_24.txt",
 channels, layer_24_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 25,layer_25_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 26, layer_26_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_26.txt",
 channels, layer_26_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 28,layer_28_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 29, layer_29_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 30,layer_30_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 31, layer_31_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 33, layer_33_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v1/fms/ifms_15.txt",
 channels, layer_15_dw_specs);
#endif
#if DEBUGGING
 verify_fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_15.txt",
 channels, layer_15_dw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 15,layer_15_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_15.txt",
 result, layer_15_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 16, layer_16_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_16.txt",
 channels, layer_16_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 17,layer_17_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_17.txt",
 result, layer_17_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 18, layer_18_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 19,layer_19_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 20, layer_20_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_20.txt",
 channels, layer_20_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 21,layer_21_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_21.txt",
 result, layer_21_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 22, layer_22_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 23,layer_23_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 24, layer_24_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_24.txt",
 channels, layer_24_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 25,layer_25_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 26, layer_26_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_26.txt",
 channels, layer_26_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 28,layer_28_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 29, layer_29_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 30,layer_30_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 31, layer_31_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 33, layer_33_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v1/fms/ifms_15.txt",
 channels, layer_15_dw_specs);
#endif
#if DEBUGGING
 verify_fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_15.txt",
 channels, layer_15_dw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 15,layer_15_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_15.txt",
 result, layer_15_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 16, layer_16_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_16.txt",
 channels, layer_16_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 17,layer_17_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_17.txt",
 result, layer_17_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 18, layer_18_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 19,layer_19_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 20, layer_20_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_20.txt",
 channels, layer_20_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 21,layer_21_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_21.txt",
 result, layer_21_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 22, layer_22_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 23,layer_23_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 24, layer_24_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_24.txt",
 channels, layer_24_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 25,layer_25_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 26, layer_26_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_26.txt",
 channels, layer_26_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 28,layer_28_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 29, layer_29_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 30,layer_30_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 31, layer_31_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 33, layer_33_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v1/fms/ifms_15.txt",
 channels, layer_15_dw_specs);
#endif
#if DEBUGGING
 verify_fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_15.txt",
 channels, layer_15_dw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 15,layer_15_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_15.txt",
 result, layer_15_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 16, layer_16_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_16.txt",
 channels, layer_16_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 17,layer_17_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_17.txt",
 result, layer_17_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 18, layer_18_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 19,layer_19_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 20, layer_20_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_20.txt",
 channels, layer_20_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 21,layer_21_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_21.txt",
 result, layer_21_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 22, layer_22_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 23,layer_23_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 24, layer_24_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_24.txt",
 channels, layer_24_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 25,layer_25_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 26, layer_26_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_26.txt",
 channels, layer_26_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 28,layer_28_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 29, layer_29_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 30,layer_30_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 31, layer_31_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 33, layer_33_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 8, layer_8_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 10,layer_10_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 11, layer_11_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v1/fms/ifms_12.txt",
 result, layer_12_dw_specs);
#endif
#if DEBUGGING
 verify_fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_12.txt",
 result, layer_12_dw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 12,layer_12_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_12.txt",
 channels, layer_12_dw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 13, layer_13_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 15,layer_15_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_15.txt",
 channels, layer_15_dw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 16, layer_16_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_16.txt",
 result, layer_16_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 17,layer_17_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_17.txt",
 channels, layer_17_dw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 18, layer_18_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 19,layer_19_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 20, layer_20_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_20.txt",
 result, layer_20_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 21,layer_21_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_21.txt",
 channels, layer_21_dw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 22, layer_22_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 23,layer_23_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 24, layer_24_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_24.txt",
 result, layer_24_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 25,layer_25_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 26, layer_26_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_26.txt",
 result, layer_26_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 28,layer_28_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 29, layer_29_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 30,layer_30_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
        model_configs_list);
pw_conv(off_chip_weights, channels , result, tmp_channels, 31, layer_31_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);
pw_conv(off_chip_weights, result , channels, tmp_channels, 33, layer_33_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2,
    model_configs_list);

