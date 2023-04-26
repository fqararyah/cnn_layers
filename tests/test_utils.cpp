#include "test_utils.h"

void fill_layer_input_from_file(string file_name, int input_size)
{
}

void dump_layer_output_no_tiling(string file_name, fms_dt ofms[max_fms_size],
								 int ofms_size, const int ofms_h, const int ofms_w)
{
	ofstream myfile;
	myfile.open(file_name);
	for (int i = 0; i < ofms_size; i++)
	{
		// if(ofms_size == 75264)cout<< to_print_ofms[i]<<"\n";
		myfile << ofms[i] << "\n";
	}
	myfile.close();
}

void dump_layer_output(string file_name, fms_dt ofms[max_fms_size],
					   const layer_specs layer_specs_struct)
{
	const int ofms_h = layer_specs_struct.layer_ofm_height;
	const int ofms_w = layer_specs_struct.layer_ofm_width;
	const int ofms_hw = ofms_h * ofms_w;
	const int ofms_size = ofms_hw * layer_specs_struct.layer_num_fils;
	const int num_of_tiles_w = layer_specs_struct.layer_num_of_ofm_tiles_w;
	const int num_of_tiles_hw = layer_specs_struct.layer_num_of_ofm_tiles_h * num_of_tiles_w;

	ofstream myfile;
	fms_dt to_print_ofms[max_fms_size];

	const int size = 32 * 32 * 192;
	bool skip_vals[size];

	int scaled_ofms_w = num_of_tiles_w * pw_tile_w;
	int scaled_ofms_h = layer_specs_struct.layer_num_of_ofm_tiles_h * pw_tile_h;

	int num_tiles_w = (scaled_ofms_w / pw_tile_w);
	int scaled_ofms_size = (ofms_size / (ofms_h * ofms_w)) * scaled_ofms_h * scaled_ofms_w;

	for (int i = 0; i < scaled_ofms_size; i++)
	{
		// if(ofms_size == 75264)cout<< ofms[i]<<" ofm\n";
		int tile_indx = i / pw_tile_size;
		int in_tile_index = i % pw_tile_size;

		int tile_in_d = tile_indx / (num_of_tiles_hw);
		int tile_in_h = (tile_indx % num_of_tiles_hw) / num_tiles_w;
		int tile_in_w = tile_indx % num_tiles_w;

		int in_tile_d = in_tile_index / pw_tile_hw;
		int in_tile_h = (in_tile_index % pw_tile_hw) / pw_tile_w;
		int in_tile_w = in_tile_index % pw_tile_w;

		int actual_index = (tile_in_d * pw_tile_d + in_tile_d) * (scaled_ofms_h * scaled_ofms_w) + (tile_in_h * pw_tile_h + in_tile_h) * scaled_ofms_w + tile_in_w * pw_tile_w + in_tile_w;

		if (actual_index < size)
		{
			if (tile_in_w * pw_tile_w + in_tile_w >= ofms_w || tile_in_h * pw_tile_h + in_tile_h >= ofms_h)
			{
				skip_vals[actual_index] = 1;
			}
			else
			{
				skip_vals[actual_index] = 0;
			}
		}
		to_print_ofms[actual_index] = ofms[i];
		// if(ofms_size == 75264)cout<< to_print_ofms[actual_index]<<" to_print_ofms"<<actual_index<<"xx\n";
	}

	myfile.open(file_name);
	for (int i = 0; i < scaled_ofms_size; i++)
	{
		if (i >= size || !skip_vals[i])
		{
			// if(ofms_size == 75264)cout<< to_print_ofms[i]<<"\n";
			myfile << (int)to_print_ofms[i] << "\n";
		}
	}
	myfile.close();
}

void dump_layer_output(string file_name, fms_dt ofms[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
					   const layer_specs layer_specs_struct)
{

	ofstream myfile;

	const int ofms_h = layer_specs_struct.layer_ofm_height;
	const int ofms_w = layer_specs_struct.layer_ofm_width;
	const int ofms_hw = ofms_h * ofms_w;
	const int ofms_size = ofms_hw * layer_specs_struct.layer_num_fils;
	const int num_of_tiles_w = layer_specs_struct.layer_num_of_ofm_tiles_w;
	const int num_of_tiles_hw = layer_specs_struct.layer_num_of_ofm_tiles_h * num_of_tiles_w;

	myfile.open(file_name);
	for (int i = 0; i < ofms_size; i++)
	{
		const int d_in_ofms = i / ofms_hw;
		const int h_in_ofms = (i % ofms_hw) / ofms_w;
		const int w_in_ofms = i % ofms_w;

		const int d = d_in_ofms * num_of_tiles_hw + (h_in_ofms / pw_tile_h) * num_of_tiles_w + w_in_ofms / pw_tile_w;
		const int h = h_in_ofms % pw_tile_h;
		const int w = w_in_ofms % pw_tile_w;

		myfile << (int)ofms[d][h][w] << "\n";
	}
	myfile.close();
}

void dump_pw_pss_tile(string file_name,
					  pss_dt tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w])
{
	ofstream myfile;
	myfile.open(file_name);

	for (int i = 0; i < pw_conv_parallelism_out; i++)
	{
		for (int j = 0; j < pw_tile_h; j++)
		{
			for (int k = 0; k < pw_tile_w; k++)
			{
				myfile << tile[i][j][k] << "\n";
			}
		}
	}

	myfile.close();
}

void dump_pw_channels_tile(string file_name,
						   fms_dt tile[pw_tile_d][pw_tile_h][pw_tile_w])
{
	ofstream myfile;
	myfile.open(file_name);

	for (int i = 0; i < pw_tile_d; i++)
	{
		for (int j = 0; j < pw_tile_h; j++)
		{
			for (int k = 0; k < pw_tile_w; k++)
			{
				myfile << tile[i][j][k] << "\n";
			}
		}
	}

	myfile.close();
}

void dump_pw_weights_tile(string file_name,
						  weights_dt tile[pw_conv_parallelism_out][max_conv_d], int layer_depth)
{
	ofstream myfile;
	myfile.open(file_name);

	for (int i = 0; i < pw_conv_parallelism_out; i++)
	{
		for (int j = 0; j < layer_depth; j++)
		{
			myfile << tile[i][j] << "\n";
		}
	}

	myfile.close();
}

void dump_ouput(string file_name, fms_dt out[], int size)
{
	ofstream myfile;
	myfile.open(file_name);

	for (int i = 0; i < size; i++)
	{
		myfile << out[i] << "\n";
	}
	myfile.close();
}

string get_model_prefix()
{
	string model_name;
	switch (MODEL_ID)
	{
	case MOB_V2:
		model_name = "mob_v2";
		break;
	case RESNET50:
		model_name = "resnet50";
	default:
		break;
	}
	return model_name;
}