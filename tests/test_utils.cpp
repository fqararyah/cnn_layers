#include "test_utils.h"

void fill_layer_input_from_file(string file_name, int input_size) {

}

void dumb_layer_output(string file_name, fms_dt ofms[max_fms_size],
		int ofms_size, const int ofms_h, const int ofms_w) {
	ofstream myfile;
	fms_dt to_print_ofms[max_fms_size];
	int num_tiles_hw = (ofms_w / pw_tile_w) * (ofms_h / pw_tile_h);
	int num_tiles_w = (ofms_w / pw_tile_w);
	for (int i = 0; i < ofms_size; i++) {
		int tile_indx = i / pw_tile_size;
		int in_tile_index = i % pw_tile_size;

		int tile_in_d = tile_indx / (num_tiles_hw);
		int tile_in_h = (tile_indx % num_tiles_hw) / num_tiles_w;
		int tile_in_w = tile_indx % num_tiles_w;

		int in_tile_d = in_tile_index / pw_tile_hw;
		int in_tile_h = (in_tile_index % pw_tile_hw) / pw_tile_w;
		int in_tile_w = in_tile_index % pw_tile_w;

		int actual_index = (tile_in_d * pw_tile_d + in_tile_d)
				* (ofms_h * ofms_w)
				+ (tile_in_h * pw_tile_h + in_tile_h) * ofms_w
				+ tile_in_w * pw_tile_w + in_tile_w;

		to_print_ofms[actual_index] = ofms[i];

	}

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
