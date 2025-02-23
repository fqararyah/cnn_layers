#include "../../basic_defs/basic_defs_glue.h"
#include "mob_v2_layers_specs.h"
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE && MODEL_ID == MOB_V2 && PIPELINE_LENGTH == 0
#ifndef CONV_PW_WEIGHTS
#define CONV_PW_WEIGHTS
const static layer_0_weights_dt first_layer_weights[first_conv_layer_num_fils][first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim]= {
{
{
{-32, -53, -13},
{52, 71, 32},
{-18, 1, -17},
},
{
{-59, -86, -19},
{96, 127, 41},
{-31, -5, -19},
},
{
{-20, -24, -11},
{30, 37, 13},
{-10, 0, -8},
},
},
{
{
{-55, 58, -9},
{-77, 64, 16},
{-5, 22, -17},
},
{
{-100, 105, -14},
{-127, 108, 25},
{-2, 30, -31},
},
{
{-30, 35, -6},
{-33, 29, 6},
{-2, 13, -12},
},
},
{
{
{31, -69, -89},
{71, -57, -127},
{93, 2, -79},
},
{
{-100, 7, 54},
{-61, 57, 108},
{-50, 32, 77},
},
{
{57, 61, 32},
{-11, 5, 24},
{-41, -33, 9},
},
},
{
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
},
{
{
{43, 29, -6},
{26, 26, 10},
{-1, 12, 10},
},
{
{-120, -112, 1},
{-101, -127, -27},
{-19, -32, -21},
},
{
{79, 89, 7},
{78, 103, 17},
{20, 20, 11},
},
},
{
{
{-30, -4, -1},
{-80, 0, -4},
{-48, 14, 1},
},
{
{-64, -10, 0},
{-127, 3, -3},
{-59, 19, -6},
},
{
{-17, -3, 3},
{-36, 1, 3},
{-19, 4, -3},
},
},
{
{
{8, -26, 3},
{-9, -114, -9},
{-9, -4, 0},
},
{
{-19, -40, 10},
{-31, -113, -7},
{-9, -33, 16},
},
{
{39, -8, 27},
{-10, -127, -15},
{12, -11, 2},
},
},
{
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
},
{
{
{-2, 18, -10},
{42, 49, 29},
{-39, -76, -14},
},
{
{0, 42, -17},
{67, 83, 44},
{-75, -127, -31},
},
{
{4, 12, -1},
{17, 23, 12},
{-26, -35, -13},
},
},
{
{
{46, 77, 32},
{-49, -57, -28},
{0, 3, -3},
},
{
{80, 127, 43},
{-84, -108, -41},
{5, 13, -4},
},
{
{19, 34, 9},
{-21, -25, -8},
{0, 3, -2},
},
},
{
{
{-116, 4, 21},
{-3, 7, 3},
{23, -4, -1},
},
{
{-127, 8, -4},
{-2, 3, -7},
{-4, -5, 6},
},
{
{-50, 16, 3},
{-9, 8, 5},
{-5, 2, 1},
},
},
{
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
},
{
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
},
{
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
},
{
{
{30, 21, 10},
{21, 92, 22},
{-15, 4, 1},
},
{
{34, 89, 68},
{29, 127, 4},
{-7, -14, -8},
},
{
{1, 11, 8},
{10, 43, 7},
{5, -3, -1},
},
},
{
{
{-19, 34, 5},
{-41, 73, -8},
{-32, 53, -12},
},
{
{-29, 50, 4},
{-63, 127, -8},
{-42, 71, -18},
},
{
{-5, 11, 3},
{-18, 32, 4},
{-9, 21, -4},
},
},
{
{
{89, 88, 87},
{91, 88, 87},
{90, 88, 85},
},
{
{124, 121, 120},
{127, 123, 121},
{127, 123, 120},
},
{
{69, 65, 63},
{73, 68, 65},
{74, 70, 65},
},
},
{
{
{-11, -35, 39},
{-19, -46, 68},
{-17, -25, 41},
},
{
{-17, -68, 76},
{-23, -95, 127},
{-20, -43, 57},
},
{
{-6, -18, 21},
{-8, -24, 36},
{-4, -11, 22},
},
},
{
{
{20, -17, 1},
{1, -21, 0},
{7, 11, -3},
},
{
{-73, -21, 77},
{-29, -47, 28},
{39, 0, -6},
},
{
{-127, -105, -93},
{-88, -65, -50},
{-45, -15, 21},
},
},
{
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
},
{
{
{-1, -67, -98},
{-32, -89, -127},
{2, -60, -112},
},
{
{115, 95, 83},
{80, 66, 47},
{42, 23, -12},
},
{
{-68, 4, 58},
{-43, 22, 66},
{-19, 30, 55},
},
},
{
{
{-26, -108, -92},
{-85, -127, -35},
{-54, -29, 49},
},
{
{16, 85, 61},
{64, 101, 28},
{38, 30, -39},
},
{
{11, 26, 30},
{25, 24, 8},
{16, -2, -8},
},
},
{
{
{-32, 64, -40},
{19, -58, 44},
{14, -7, -9},
},
{
{-68, 90, -31},
{99, -127, 40},
{-32, 38, -9},
},
{
{-19, 13, 5},
{31, -36, 5},
{-14, 22, -8},
},
},
{
{
{5, 38, 75},
{-36, -14, 26},
{-60, -37, 5},
},
{
{1, -7, -3},
{10, -1, 10},
{44, 45, 61},
},
{
{31, 78, 127},
{-47, -15, 35},
{-108, -73, -28},
},
},
{
{
{-84, -78, -7},
{-50, -43, 8},
{-5, -4, 6},
},
{
{-124, -127, -3},
{-59, -68, -6},
{-1, 3, 0},
},
{
{-45, -39, -11},
{-22, -23, -7},
{1, -2, 4},
},
},
{
{
{4, 11, 2},
{-36, 8, -6},
{-80, 15, -6},
},
{
{13, 20, 9},
{-74, 16, -8},
{-127, 32, -4},
},
{
{2, 5, 1},
{-18, 3, -1},
{-34, 11, -1},
},
},
{
{
{-45, -45, -45},
{-48, -48, -49},
{-49, -50, -50},
},
{
{-68, -67, -62},
{-70, -71, -66},
{-72, -73, -68},
},
{
{-120, -120, -115},
{-123, -123, -119},
{-126, -127, -122},
},
},
{
{
{-19, 66, -48},
{4, 77, -78},
{-27, 30, -3},
},
{
{-32, 117, -82},
{3, 127, -123},
{-46, 36, 7},
},
{
{-12, 42, -32},
{6, 38, -40},
{-17, 18, -3},
},
},
{
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
{
{0, 0, 0},
{0, 0, 0},
{0, 0, 0},
},
},
{
{
{-1, -13, 41},
{-127, -124, -34},
{-114, -100, -65},
},
{
{1, -2, -46},
{66, 46, -2},
{56, 42, 27},
},
{
{3, 19, 9},
{59, 82, 39},
{53, 64, 32},
},
},
{
{
{74, 20, -8},
{-5, 5, -9},
{-14, 1, -13},
},
{
{127, 0, -19},
{0, -19, 0},
{-1, 12, -20},
},
{
{45, -14, -3},
{10, 5, 12},
{-1, -1, -7},
},
},
{
{
{59, 10, -123},
{123, 53, 19},
{98, -3, -37},
},
{
{19, 127, 53},
{-84, -86, -2},
{-6, 37, 10},
},
{
{-35, 36, 31},
{-46, -32, 36},
{-118, -3, 16},
},
},
};
#endif
#endif
