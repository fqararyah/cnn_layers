#include "test_utils.h"

void fill_layer_input_from_file(string file_name, int input_size) {

}

void dumb_layer_output(string file_name, fms_dt ofms[max_fms_size],
		int ofms_size) {
	ofstream myfile;
	myfile.open(file_name);
	for (int i = 0; i < ofms_size; i++) {
		myfile << ofms[i] << "\n";
	}
	myfile.close();
}

void dumb_pw_pss_tile(string file_name,
		pss_dt tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w]) {
	ofstream myfile;
	myfile.open(file_name);

	for (int i = 0; i < pw_conv_parallelism_out; i++) {
		for (int j = 0; j < pw_tile_h; j++) {
			for (int k = 0; k < pw_tile_w; k++) {
				myfile << tile[i][j][k] << "\n";
			}
		}
	}

	myfile.close();
}


void dumb_pw_channels_tile(string file_name,
		fms_dt tile[pw_tile_d][pw_tile_h][pw_tile_w]) {
	ofstream myfile;
	myfile.open(file_name);

	for (int i = 0; i < pw_tile_d; i++) {
		for (int j = 0; j < pw_tile_h; j++) {
			for (int k = 0; k < pw_tile_w; k++) {
				myfile << tile[i][j][k] << "\n";
			}
		}
	}

	myfile.close();
}

void dumb_pw_weights_tile(string file_name,
		weights_dt tile[pw_conv_parallelism_out][max_conv_d], int layer_depth) {
	ofstream myfile;
	myfile.open(file_name);

	for (int i = 0; i < pw_conv_parallelism_out; i++) {
		for (int j = 0; j < layer_depth; j++) {
			myfile << tile[i][j] << "\n";
		}
	}

	myfile.close();
}
