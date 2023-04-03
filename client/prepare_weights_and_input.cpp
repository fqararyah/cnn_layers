#include "prepare_weights_and_inputs.h"

bool isNumber(string &str)
{
	for (char const &c : str)
	{
		if (std::isdigit(c) == 0)
			return false;
	}
	return true;
}

int get_num_of_pw_weights(string file_name)
{
	std::ifstream infile(file_name);
	assert(!infile.fail());
	int num_of_pw_weights = -1;
	while (infile >> num_of_pw_weights)
		;
	return num_of_pw_weights;
}

void load_weights(string file_name,
				  weights_dt weights[])
{
	int a;
	std::ifstream infile(file_name);
	assert(!infile.fail());
	int line_num = 0;
	while (infile >> a)
	{
		weights[line_num] = a;
		line_num++;
	}
}

void load_image(string file_name,
				fms_dt image[])
{
	int a;
	std::ifstream infile(file_name);
	assert(!infile.fail());
	int line_num = 0;
	while (infile >> a)
	{
		image[line_num] = a;
		line_num++;
	}
}

#if FPGA
void glue_weights(string file_name,
				  weights_grp_dt glued_weights[all_pw_weights])
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
		glued_weights[external_index](
			internal_index * weights_dt_width + weights_dt_offset,
			internal_index * weights_dt_width) = weight;
		line_num++;
	}
}

void validate_weights(string file_name,
					  weights_grp_dt glued_weights[all_pw_weights])
{
	int a;
	std::ifstream infile(file_name);
	assert(!infile.fail());
	bool failed = false;
	int line_num = 0;
	while (infile >> a)
	{
		weights_dt weight = (weights_dt)a;
		int external_index = line_num / weights_group_items;
		int internal_index = line_num % weights_group_items;
		if (weight != (weights_dt)glued_weights[external_index](
						  internal_index * weights_dt_width + weights_dt_offset,
						  internal_index * weights_dt_width))
		{
			cout << "failed at: " << line_num << " " << weight << " != "
				 << (weights_dt)glued_weights[external_index](
						internal_index * weights_dt_width + weights_dt_offset,
						internal_index * weights_dt_width);
			failed = true;
			break;
		}
		line_num++;
	}
	if (!failed)
	{
		cout << line_num << " weights have been glued SUCESSFULLY!\n";
	}
}

void glue_input_image(string file_name,
					  fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel])
{
	int a;
	std::ifstream infile(file_name);
	assert(!infile.fail());

	int line_num = 0;
	while (infile >> a)
	{
		fms_dt val = (fms_dt)a;
		const int z = line_num / input_image_hw;
		const int y = (line_num % input_image_hw) / input_image_width;
		const int x = line_num % input_image_width;
		int external_index = z * input_image_num_fms_groups_in_a_channel + y * input_image_num_fms_groups_in_width + x / input_image_group_items;
		int internal_index = x % input_image_group_items;
		//		cout << line_num << " >> " << z << ", " << y << ", " << x << ", "
		//				<< external_index << ", " << internal_index << "\n";
		input_image[external_index](
			internal_index * fms_dt_width + fms_dt_offset,
			internal_index * fms_dt_width) = val;
		line_num++;
	}
}

void verify_glued_image(string file_name,
						fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel])
{
	int a;
	std::ifstream infile(file_name);
	assert(!infile.fail());
	bool failed = false;
	int line_num = 0;
	while (infile >> a)
	{
		fms_dt val = (fms_dt)a;
		const int z = line_num / input_image_hw;
		const int y = (line_num % input_image_hw) / input_image_width;
		const int x = line_num % input_image_width;
		int external_index = z * input_image_num_fms_groups_in_a_channel + y * input_image_num_fms_groups_in_width + x / input_image_group_items;
		int internal_index = x % input_image_group_items;
		if (val != (fms_dt)input_image[external_index](
					   internal_index * fms_dt_width + fms_dt_offset,
					   internal_index * fms_dt_width))
		{
			cout << "\n failed at: " << line_num << " " << val << " != "
				 << (fms_dt)input_image[external_index](
						internal_index * fms_dt_width + fms_dt_offset,
						internal_index * fms_dt_width)
				 << "\n";
			failed = true;
			break;
		}
		line_num++;
	}
	if (!failed)
	{
		cout << "\n"
			 << line_num
			 << " input image items have been glued SUCESSFULLY!\n";
	}
}
#endif

void fill_input_image(string file_name,
					  fms_dt input_image[input_image_depth][input_image_height][input_image_width])
{
	int a;
	std::ifstream infile(file_name);
	assert(!infile.fail());
	bool failed = false;
	int line_num = 0;
	const int input_image_hw = input_image_height * input_image_width;
	while (infile >> a)
	{
		int channel_index = line_num / input_image_hw;
		int channel_row = (line_num % input_image_hw) / input_image_width;
		int channel_col = line_num % input_image_width;
		line_num++;
		input_image[channel_index][channel_row][channel_col] = (fms_dt)a;
	}
}

void verify_input_image(string file_name,
						fms_dt input_image[input_image_depth][input_image_height][input_image_width])
{

	ofstream myfile;
	const int input_image_hw = input_image_height * input_image_width;
	myfile.open(file_name);

	for (int d = 0; d < input_image_depth; d++)
	{
		for (int h = 0; h < input_image_height; h++)
		{
			for (int w = 0; w < input_image_width; w++)
			{
				myfile << input_image[d][h][w] << "\n";
			}
		}
	}
	myfile.close();
}

void fill_layer_input(string file_name, fms_dt layer_input[max_fms_size],
					  const int ifms_h, const int ifms_w)
{

	int ofms_hw = ifms_h * ifms_w;
	int num_tiles_h =
		(ifms_h % pw_tile_h) == 0 ? (ifms_h / pw_tile_h) : (ifms_h / pw_tile_h) + 1;
	int num_tiles_w =
		(ifms_w % pw_tile_w) == 0 ? (ifms_w / pw_tile_w) : (ifms_w / pw_tile_w) + 1;
	int num_tiles_hw = num_tiles_h * num_tiles_w;

	int a;
	int line = 0;
	std::ifstream infile(file_name);
	assert(!infile.fail());
	while (infile >> a)
	{
		int z = (line / ofms_hw);
		int h = ((line % ofms_hw) / ifms_w);
		int w = (line % ifms_w);

		int tile_in_z = z / pw_tile_d;
		int tile_in_h = h / pw_tile_h;
		int tile_in_w = w / pw_tile_w;
		int tile_index = tile_in_z * num_tiles_hw + tile_in_h * num_tiles_w + tile_in_w;

		int in_tile_z = z % pw_tile_d;
		int in_tile_h = h % pw_tile_h;
		int in_tile_w = w % pw_tile_w;
		int in_tile_index = in_tile_z * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;

		int actual_index = tile_index * pw_tile_size + in_tile_index;

		layer_input[actual_index] = (fms_dt)a;
		line++;
	}
}

void verify_fill_layer_input(string file_name, fms_dt ifms[max_fms_size],
							 const int ifms_size, const int ifms_h, const int ifms_w)
{

	ofstream myfile;
	fms_dt to_print_ofms[max_fms_size];
	int num_tiles_h =
		(ifms_h % pw_tile_h) == 0 ? (ifms_h / pw_tile_h) : (ifms_h / pw_tile_h) + 1;
	int num_tiles_w =
		(ifms_w % pw_tile_w) == 0 ? (ifms_w / pw_tile_w) : (ifms_w / pw_tile_w) + 1;
	int num_tiles_hw = num_tiles_h * num_tiles_w;

	for (int i = 0; i < ifms_size; i++)
	{
		int tile_indx = i / pw_tile_size;
		int in_tile_index = i % pw_tile_size;

		int tile_in_d = tile_indx / (num_tiles_hw);
		int tile_in_h = (tile_indx % num_tiles_hw) / num_tiles_w;
		int tile_in_w = tile_indx % num_tiles_w;

		int in_tile_d = in_tile_index / pw_tile_hw;
		int in_tile_h = (in_tile_index % pw_tile_hw) / pw_tile_w;
		int in_tile_w = in_tile_index % pw_tile_w;

		int actual_index = (tile_in_d * pw_tile_d + in_tile_d) * (ifms_h * ifms_w) + (tile_in_h * pw_tile_h + in_tile_h) * ifms_w + tile_in_w * pw_tile_w + in_tile_w;

		to_print_ofms[actual_index] = ifms[i];
	}
	myfile.open(file_name);
	for (int i = 0; i < ifms_size; i++)
	{
		myfile << (int)to_print_ofms[i] << "\n";
	}
	myfile.close();
}
