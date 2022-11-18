
#ifndef SESL_SPECS
#define SESL_SPECS

//*****variable based on experiment*****
const int sesl_layer_0_parallelism_ofms = 1;

const int layer_1_pw_parallelism_in = 32;
const int layer_1_pw_parallelism_out = 1;
const int layer_3_pw_parallelism_in = 32;
const int layer_3_pw_parallelism_out = 1;
const int layer_4_pw_parallelism_in = 16;
const int layer_4_pw_parallelism_out = 2;
const int layer_6_pw_parallelism_in = 96;
const int layer_6_pw_parallelism_out = 1;
const int layer_7_pw_parallelism_in = 24;
const int layer_7_pw_parallelism_out = 1;

const int layer_1_dw_parallelism = layer_1_pw_parallelism_out;
const int sesl_layer_2_dw_parallelism = 1;
const int layer_3_dw_parallelism = layer_3_pw_parallelism_out;

const int layer_1_pw_parallelism_in_input_partitioning_factor =
	layer_1_pw_parallelism_in / 2 >= sesl_layer_0_parallelism_ofms ? layer_1_pw_parallelism_in / 2 : sesl_layer_0_parallelism_ofms;
const int dw_layer_2_parallelism_input_partitioning_factor =
	sesl_layer_2_dw_parallelism / 2 >= layer_1_pw_parallelism_out ? sesl_layer_2_dw_parallelism / 2 : layer_1_pw_parallelism_out;
const int dw_layer_3_parallelism_input_partitioning_factor =
	layer_3_dw_parallelism / 2;

#endif
