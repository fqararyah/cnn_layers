#include "fpga_prepare_weights_and_inputs.h"

using namespace pipelined_engines;

#if HW == CPU
void fill_pipe_layer_input_buffer(string file_name, fms_dt channels_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
								  const int starting_h, const int start_filling_offset_in_buffer,
								  const layer_specs layer_specs_struct)
{

	const int ifms_h = layer_specs_struct.layer_ifm_height;
	const int ifms_w = layer_specs_struct.layer_ifm_width;
	const int ifms_hw = ifms_h * ifms_w;

	int a;
	int line = 0;
	std::ifstream infile(file_name);
	assert(!infile.fail());
	while (infile >> a)
	{
		int d = line / ifms_hw;
		int h = ((line % ifms_hw) / ifms_w) - starting_h;
		int w = line % ifms_w;
		line++;
		if (h < 0 || h >= PW_BUFFER_HEIGHT || start_filling_offset_in_buffer + h >= PW_BUFFER_HEIGHT)
		{
			continue;
		}
		channels_buffer[d][h + start_filling_offset_in_buffer][w] = (fms_dt)a;
	}
}
#endif

void fill_layer_input(string file_name, fms_dt layer_input[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
					  const layer_specs layer_specs_struct)
{
	const int ifms_h = layer_specs_struct.layer_ifm_height;
	const int ifms_w = layer_specs_struct.layer_ifm_width;
	const int ifms_hw = ifms_h * ifms_w;
	const int num_of_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_w;
	const int num_of_tiles_hw = layer_specs_struct.layer_num_of_ifm_tiles_h * num_of_tiles_w;

	int a;
	int line = 0;
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
							 const layer_specs layer_specs_struct)
{
	ofstream myfile;

	const int ifms_h = layer_specs_struct.layer_ifm_height;
	const int ifms_w = layer_specs_struct.layer_ifm_width;
	const int ifms_hw = ifms_h * ifms_w;
	const int ifms_size = ifms_hw * layer_specs_struct.layer_depth;
	const int num_of_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_w;
	const int num_of_tiles_hw = layer_specs_struct.layer_num_of_ifm_tiles_h * num_of_tiles_w;

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

void glue_on_chip_weights_cpu(string file_name,
							  weights_grp_dt glued_on_chip_weights[all_on_chip_pw_s_weights_groups])
{
	int a;
	std::ifstream infile(file_name);
	assert(!infile.fail());
	int line_num = 0;
	while (infile >> a)
	{
		weights_grp_dt weight = (weights_grp_dt)a;
		glued_on_chip_weights[line_num] = weight;
		line_num++;
	}
}

#if HW == _FPGA

void glue_on_chip_weights_fpga(string file_name,
							   weights_grp_dt glued_on_chip_weights[all_on_chip_pw_s_weights_groups])
{
	int a;
	std::ifstream infile(file_name);
	assert(!infile.fail());
	int line_num = 0;
	while (infile >> a)
	{
		weights_dt weight = (weights_dt)a;
		int external_index = line_num / weights_group_items;
		int internal_index = line_num % weights_group_items;
		glued_on_chip_weights[external_index](
			internal_index * weights_dt_width + weights_dt_offset,
			internal_index * weights_dt_width) = weight;
		line_num++;
	}
}
#endif
