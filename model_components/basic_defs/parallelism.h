
#ifndef PARALLELISM_HEADER
#define PARALLELISM_HEADER

const int PW_PARALLELISM_H = 4;
const int PW_PARALLELISM_W = 4;
const int DW_PARALLELISM_H = PW_PARALLELISM_H;
const int DW_PARALLELISM_W = PW_PARALLELISM_W;

const int pw_conv_parallelism_in = 1;
// WARNING, when pw_conv_parallelism_out is changes, generate script should be run
const int pw_conv_parallelism_out = 8; //>= tile_d and >=8: 16, 32, 64 (< 8 is not working for weight load)

#endif
