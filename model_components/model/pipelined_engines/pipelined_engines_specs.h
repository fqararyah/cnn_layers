#ifndef PIPELINED_ENGINES_SPECS
#define PIPELINED_ENGINES_SPECS

namespace pipelined_engines
{
    const int PARALLELISM_PW_OFMS = 24;
    const int PARALLELISM_PW_IFMS = 0; // no parallelism
    const int PARALLELISM_PW_H = 4;
    const int PARALLELISM_PW_W = 2;

    const int MAX_PW_BUFFER_DEPTH = 144;
    const int MAX_PW_BUFFER_WIDTH = 112;
    const int PW_BUFFER_HEIGHT = PARALLELISM_PW_H;
    const int PW_BUFFER_WIDTH = 2;

    const int PIPELINE_TMP_CHANNELS_HEIGHT = PW_BUFFER_HEIGHT + 1;

    const int PARALLELISM_DW_OFMS = 0;
    const int PARALLELISM_DW_IFMS = 0;
    const int PARALLELISM_DW_H = PARALLELISM_PW_H;
    const int PARALLELISM_DW_W = PW_BUFFER_WIDTH;

    const int MAX_DW_FILTER_DIM_IN_PIPE = 3;
    const int MAX_DW_STRIDES_IN_PIPE = 2;
    const int MAX_DW_PADDING_IN_PIPE = 1;
    const int MAX_FILTER_MINUS_STRIDES = 3 - 1;
    const int MIN_FILTER_MINUS_STRIDES = 3 - 2;
    const int MAX_DW_FILTER_AREA_IN_PIPE = MAX_DW_FILTER_DIM_IN_PIPE * MAX_DW_FILTER_DIM_IN_PIPE;

    const int DW_BUFFER_DEPTH = PARALLELISM_PW_OFMS;
    const int DW_BUFFER_HEIGHT = PW_BUFFER_HEIGHT + MAX_DW_FILTER_DIM_IN_PIPE - 1;
    const int DW_BUFFER_WIDTH = PW_BUFFER_WIDTH + 2 * MAX_DW_PADDING_IN_PIPE;

    const int DW_TILE_DEPTH = PARALLELISM_PW_OFMS;

    const int DW_PIPE_OVERLAP_BUFFER_DEPTH = 4 * 32 + 2 * 96 + 2 * 144 + 144;
    const int DW_PIPE_OVERLAP_BUFFER_WIDTH = 56;

    const int OFFSET_H_IN_RESULTS = PW_BUFFER_HEIGHT / 2;

    const int STRIDES_PRODUCT_IN_PIPELINE = 8; // TODO automate

    const int PRE_FIRST_PIPELINE_OUTPUT_DEPTH = layer_2_dw_depth;
    const int PRE_FIRST_PIPELINE_OUTPUT_HEIGHT = 8;//TODO
    const int PRE_FIRST_PIPELINE_OUTPUT_WIDTH = layer_2_dw_ifm_width;

    const int INPUT_IMAGE_ROWS_FILLED_EACH_TIME = first_conv_layer_strides * 1; // 1 is strides of the first dw layer
    const int FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME = first_conv_layer_filter_dim - first_conv_layer_strides;
    const int PRE_FIRST_PIPELINE_INPUT_HEIGHT = INPUT_IMAGE_ROWS_FILLED_EACH_TIME + FIRST_CONV_LAYER_EXTRA_ROWS_FILLED_FIRST_TIME;

    const int FIRST_CONV_LAYER_BUFFER_SIZE = first_conv_layer_filter_dim * first_conv_layer_filter_dim * input_image_depth;
    const int FIRST_CONV_LAYER_NEW_COLS_BUFFER_SIZE = first_conv_layer_filter_dim * first_conv_layer_strides * input_image_depth;

}
#endif