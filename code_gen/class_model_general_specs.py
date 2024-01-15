class general_model_specs:
    def __init__(self, max_conv_d = 0,
                max_filter_hw_dim = 3,
                max_std_conv_filter_hw_dim = 3,
                max_padding_lr = 0, 
                fc_layer_input_size = 0, 
                max_tmp_fms_shape = [0,0,0],
                max_fms_shape = [0,0,0],
                max_filter_dim_stride_1 = 3,
                max_filter_dim_stride_2 = 3,
                max_dw_layer_d = 0,
                all_on_chip_pw_s_weights = 0,
                all_dw_off_chip_weights = 0,
                all_off_chip_fused_scales_zps = 0,
                all_off_chip_pw_s_weights = 0):
        self.max_conv_d = max_conv_d
        self.max_filter_hw_dim = max_filter_hw_dim
        self.max_std_conv_filter_hw_dim = max_std_conv_filter_hw_dim
        self.max_padding_lr = max_padding_lr
        self.fc_layer_input_size = fc_layer_input_size
        self.max_tmp_fms_shape = max_tmp_fms_shape
        self.max_fms_shape = max_fms_shape
        self.max_filter_dim_stride_1 = max_filter_dim_stride_1
        self.max_filter_dim_stride_2 = max_filter_dim_stride_2
        self.max_dw_layer_d = max_dw_layer_d
        self.all_on_chip_pw_s_weights = all_on_chip_pw_s_weights
        self.all_dw_off_chip_weights = all_dw_off_chip_weights
        self.all_off_chip_fused_scales_zps = all_off_chip_fused_scales_zps
        self.all_off_chip_pw_s_weights = all_off_chip_pw_s_weights

    def write_specs_to_header(self, header_file_name, model_name, pipeline_len):
        with open(header_file_name, 'w') as f:
            f.write('#include "parallelism.h"\n\n')
            f.write('#if MODEL_ID == ' + model_name.upper() + ' && PIPELINE_LENGTH == ' + str(pipeline_len) + '\n\n')
            f.write('#ifndef MODEL_GENERAL_SPECS_HEADER\n')
            f.write('#define MODEL_GENERAL_SPECS_HEADER\n\n')
            f.write('const int max_conv_d = {} / alpha;\n'.format(self.max_conv_d)),
            f.write('const int max_filter_hw_dim = {};\n'.format(self.max_filter_hw_dim))
            f.write('const int max_std_conv_filter_hw_dim = {};\n'.format(self.max_std_conv_filter_hw_dim))
            f.write('const int max_padding_lr = {};\n'.format(self.max_padding_lr))
            f.write('const int fc_layer_input_size = {};\n'.format(self.fc_layer_input_size))
            f.write('const int MAX_TMP_FMS_BUFFER_DEPTH = (({} + CHANNELS_TILE_HEIGHT - 1) / CHANNELS_TILE_HEIGHT) * (({} + CHANNELS_TILE_WIDTH - 1) / CHANNELS_TILE_WIDTH) * {};\n'.format(
                                                                                            self.max_tmp_fms_shape[1],
                                                                                            self.max_tmp_fms_shape[2],
                                                                                            self.max_tmp_fms_shape[0]))
            f.write('const int MAX_FMS_BUFFER_DEPTH = (({} + CHANNELS_TILE_HEIGHT - 1) / CHANNELS_TILE_HEIGHT) * (({} + CHANNELS_TILE_WIDTH - 1) / CHANNELS_TILE_WIDTH) * {};\n'.format(
                                                                                        self.max_fms_shape[1],
                                                                                        self.max_fms_shape[2],
                                                                                        self.max_fms_shape[0]))
            f.write('const int MAX_FILTER_DIM_STRIDE_1 = {};\n'.format(self.max_filter_dim_stride_1))
            f.write('const int MAX_FILTER_DIM_STRIDE_2 = {};\n'.format(self.max_filter_dim_stride_2))
            f.write('const int MAX_DW_LAYER_D = {};\n'.format(self.max_dw_layer_d))
            f.write('const int all_on_chip_pw_s_weights = {};\n'.format(self.all_on_chip_pw_s_weights))
            f.write('const int all_dw_off_chip_weights = {};\n'.format(self.all_dw_off_chip_weights))
            f.write('const int all_off_chip_fused_scales_zps = {};\n'.format(self.all_off_chip_fused_scales_zps))
            f.write('const int all_off_chip_pw_s_weights = {} / weights_group_items;\n'.format(self.all_off_chip_pw_s_weights))
            
            f.write("\n#endif\n")
            f.write("\n#endif\n")
            