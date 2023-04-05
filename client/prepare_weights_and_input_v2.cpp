#include "prepare_weights_and_inputs.h"

void fill_layer_input(string file_name, fms_dt layer_input[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
					  const int ifms_h, const int ifms_w, const int num_of_tiles_h, const int num_of_tiles_w)
{
	int a;
	int line = 0;
	const int ifms_hw = ifms_h * ifms_w;
	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;

	std::ifstream infile(file_name);
	assert(!infile.fail());
	while (infile >> a)
	{
		const int d_in_ifms = line / ifms_hw;
		const int h_in_ifms = (line % ifms_hw) / ifms_w;
		const int w_in_ifms = line % ifms_w;

		const int d = d_in_ifms * num_of_tiles_hw + num_of_tiles_w * (h_in_ifms / pw_tile_h) + w_in_ifms / pw_tile_w;
		const int h = h_in_ifms % pw_tile_h;
		const int w = w_in_ifms % pw_tile_w;

		layer_input[d][h][w] = (fms_dt)a;
		line++;
	}
}

void verify_fill_layer_input(string file_name, fms_dt layer_input[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
							 const int ifms_size, const int ifms_h, const int ifms_w, const int num_of_tiles_h, const int num_of_tiles_w)
{
	ofstream myfile;
	const int ifms_hw = ifms_h * ifms_w;
	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;

	myfile.open(file_name);

	for (int i = 0; i < ifms_size; i++)
	{
		const int d_in_ifms = i / ifms_hw;
		const int h_in_ifms = (i % ifms_hw) / ifms_w;
		const int w_in_ifms = i % ifms_w;

		const int d = d_in_ifms * num_of_tiles_hw + num_of_tiles_w * (h_in_ifms / pw_tile_h) + w_in_ifms / pw_tile_w;
		const int h = h_in_ifms % pw_tile_h;
		const int w = w_in_ifms % pw_tile_w;

		myfile << (int)layer_input[d][h][w] << "\n";
	}
	myfile.close();
}