#include "parallelism.h"

#if MODEL_ID == MOB_V2 && PIPELINE_LENGTH == 6

#ifndef MODEL_GENERAL_SPECS_HEADER
#define MODEL_GENERAL_SPECS_HEADER

const int max_conv_d = 960 / alpha;
const int max_num_filters = 1280 / alpha;
const int max_filter_hw_dim = 3;
const int max_std_conv_filter_hw_dim = 3;
const int max_padding_lr = 2;
const int fc_layer_input_size = 1280;
const int MAX_TMP_FMS_BUFFER_DEPTH = ((56 + CHANNELS_TILE_HEIGHT - 1) / CHANNELS_TILE_HEIGHT) * ((56 + CHANNELS_TILE_WIDTH - 1) / CHANNELS_TILE_WIDTH) * 24;
const int MAX_FMS_BUFFER_DEPTH = ((56 + CHANNELS_TILE_HEIGHT - 1) / CHANNELS_TILE_HEIGHT) * ((56 + CHANNELS_TILE_WIDTH - 1) / CHANNELS_TILE_WIDTH) * 144;
const int MAX_FILTER_DIM_STRIDE_1 = 3;
const int MAX_FILTER_DIM_STRIDE_2 = 3;
const int MAX_DW_LAYER_D = 960;
const int all_on_chip_pw_s_weights = 5216;
const int all_dw_off_chip_weights = 63072;
const int all_off_chip_fused_scales_zps = 16760;
const int all_off_chip_pw_s_weights = 2120320 / weights_group_items;

#endif

#endif
