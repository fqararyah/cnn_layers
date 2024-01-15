// #if MODEL_ID == MOB_V2 and PIPELINE_LENGTH == 0

// #ifndef MODEL_GENERAL_SPECS_HEADER
// #define MODEL_GENERAL_SPECS_HEADER

// const int MAX_TMP_FMS_BUFFER_DEPTH = 24 * 49; // 192 * (28/7) * (28/7)
// const int MAX_FMS_BUFFER_DEPTH = 96 * 196; // 192 * (28/7) * (28/7)

// const int MAX_FILTER_DIM_STRIDE_1 = 3;
// const int MAX_FILTER_DIM_STRIDE_2 = 3;
// const int MAX_DW_LAYER_D = 144;

// // fc_layer
// const int fc_layer_input_size = 1280;

// // maxs for buffers
// const int max_conv_d = 1280 / alpha; // to_automate

// const int max_filter_hw_dim = 3;
// const int max_std_conv_filter_hw_dim = 3;
// const int max_padding_lr = 2;

// // weights
// const int all_off_chip_pw_s_weights = 2124672 / weights_group_items;
// const int all_on_chip_pw_s_weights = 864;
// const int all_dw_off_chip_weights = 64224;
// const int all_off_chip_fused_scales_zps = 576;

// #endif

// #endif
