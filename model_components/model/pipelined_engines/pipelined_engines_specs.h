#ifndef PIPELINED_ENGINES_SPECS
#define PIPELINED_ENGINES_SPECS

namespace pipelined_engines
{
    const int PARALLELISM_PW_OFMS = 24;
    const int PARALLELISM_PW_IFMS = 0; // no parallelism
    const int PARALLELISM_PW_H = 4;
    const int PARALLELISM_PW_W = 1;

    const int MAX_PW_BUFFER_DEPTH = 144;
    const int MAX_PW_BUFFER_WIDTH = 112;
    const int PW_BUFFER_HEIGHT = PARALLELISM_PW_H;
    const int PW_BUFFER_WIDTH = 2;

    const int PARALLELISM_DW_OFMS = 0;
    const int PARALLELISM_DW_IFMS = 0;
    const int PARALLELISM_DW_H = PARALLELISM_PW_H;
    const int PARALLELISM_DW_W = PW_BUFFER_WIDTH;

    const int MAX_DW_FILTER_DIM_IN_PIPE = 3;
    const int MAX_DW_STRIDES_IN_PIPE = 2;
    const int MAX_DW_PADDING_IN_PIPE = 1;
    const int MAX_FILTER_MINUS_STRIDES = 3 - 1;
    const int MAX_DW_FILTER_AREA_IN_PIPE = MAX_DW_FILTER_DIM_IN_PIPE * MAX_DW_FILTER_DIM_IN_PIPE;

    const int DW_BUFFER_DEPTH = PARALLELISM_PW_OFMS;
    const int DW_BUFFER_HEIGHT = PW_BUFFER_HEIGHT + MAX_DW_FILTER_DIM_IN_PIPE - 1;
    const int DW_BUFFER_WIDTH = PW_BUFFER_WIDTH + 2 * MAX_DW_PADDING_IN_PIPE;

    const int DW_TILE_DEPTH = PARALLELISM_PW_OFMS;

    const int DW_PIPE_OVERLAP_BUFFER_DEPTH = 4 * 32 + 2 * 96 + 2 * 144 + 144;
    const int DW_PIPE_OVERLAP_BUFFER_WIDTH = 56;
}
#endif