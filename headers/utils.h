#include <stdint.h>
#include <string>
#include "cnn_functions_v1.h"

using namespace std;
const int image_depth = 32;
const int image_height = 112;
const int image_width = 112;

void _5_fill_layers_weights(layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
                            dw_weights_dt dw_weights_1[layer_1_dw_depth][layer_1_dw_filter_size][layer_1_dw_filter_size],
                            weights_dt pw_weights_1[layer_1_pw_num_fils][layer_1_pw_depth],
                            weights_dt pw_weights_2[layer_2_pw_num_fils][layer_2_pw_depth],
                            weights_dt pw_weights_3[layer_3_pw_num_fils][layer_3_pw_depth]);

void _7_fill_layers_weights(layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
                            dw_weights_dt dw_weights_1[layer_1_dw_depth][layer_1_dw_filter_size][layer_1_dw_filter_size],
                            dw_weights_dt dw_weights_3[layer_3_dw_depth][layer_3_dw_filter_size][layer_3_dw_filter_size],
                            weights_dt pw_weights_1[layer_1_pw_num_fils][layer_1_pw_depth],
                            weights_dt pw_weights_2[layer_2_pw_num_fils][layer_2_pw_depth],
                            weights_dt pw_weights_3[layer_3_pw_num_fils][layer_3_pw_depth],
                            weights_dt pw_weights_4[layer_4_pw_num_fils][layer_4_pw_depth]);

void v1_3_fill_layers_weights(layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
                         dw_weights_dt dw_weights_1[layer_1_dw_depth][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size],
                         weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth]);

void v1_4_fill_layers_weights(layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
                         dw_weights_dt dw_weights_1[layer_1_dw_depth][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size],
                         dw_weights_dt dw_weights_2[v1_layer_2_dw_depth][v1_layer_2_dw_filter_size][v1_layer_2_dw_filter_size],
                         weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth]);

void v1_7_fill_layers_weights(layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
	dw_weights_dt dw_weights_1[v1_layer_1_dw_depth][v1_layer_3_dw_filter_size][v1_layer_3_dw_filter_size],
	dw_weights_dt dw_weights_2[v1_layer_2_dw_depth][v1_layer_2_dw_filter_size][v1_layer_2_dw_filter_size],
	dw_weights_dt dw_weights_3[v1_layer_3_dw_depth][v1_layer_3_dw_filter_size][v1_layer_3_dw_filter_size],
	weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth],
	weights_dt pw_weights_3[v1_layer_3_pw_num_fils][v1_layer_3_pw_depth],
	weights_dt pw_weights_4[v1_layer_4_pw_num_fils][v1_layer_4_pw_depth]);

void fill_layer_0_weights(layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3]);

void fill_dw_layer_weights(const dw_weights_dt src[max_conv_d][max_conv_h][max_conv_w],
                            dw_weights_dt dst[max_conv_d][max_conv_h][max_conv_w], const int conv_d, const int conv_h, const int conv_w);

// void read_image(string file_name, uint8_t *image);
// void read_image(string file_name, float *image);
// void read_image_m(string file_name, float image[image_depth][image_height][image_width], bool dummy);
// void read_weights(string file_name, weights_dt weights[32][1][3][3]);
// void print_image(int width, int height, float *image);
// void print_image_m(float image[image_depth][image_height][image_width]);
//
// void read_weights(string file_name, weights_dt weights[32][9]);
// void read_weights_m(string file_name, weights_dt weights[num_fils][conv_d], bool dummy);
