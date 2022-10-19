#include "prepare_weights_and_inputs.h"

bool isNumber(string &str) {
	for (char const &c : str) {
		if (std::isdigit(c) == 0)
			return false;
	}
	return true;
}

void glue_weights(string file_name,
		weights_grp_dt glued_weights[all_pw_weights]) {
	int a;
	std::ifstream infile(file_name);

	int line_num = 0;
	while (infile >> a) {
		weights_dt weight = (weights_dt) a;
		int external_index = line_num / weights_group_items;
		int internal_index = line_num % weights_group_items;
		glued_weights[external_index](
				internal_index * weights_dt_width + weights_dt_offset,
				internal_index * weights_dt_width) = weight;
		line_num++;
	}
}

void validate_weights(string file_name,
		weights_grp_dt glued_weights[all_pw_weights]) {
	int a;
	std::ifstream infile(file_name);
	bool failed = false;
	int line_num = 0;
	while (infile >> a) {
		weights_dt weight = (weights_dt) a;
		int external_index = line_num / weights_group_items;
		int internal_index = line_num % weights_group_items;
		if (weight
				!= (weights_dt) glued_weights[external_index](
						internal_index * weights_dt_width + weights_dt_offset,
						internal_index * weights_dt_width)) {
			cout << "failed at: " << line_num << " " << weight << " != "
					<< (weights_dt) glued_weights[external_index](
							internal_index * weights_dt_width
									+ weights_dt_offset,
							internal_index * weights_dt_width);
			failed = true;
			break;
		}
		line_num++;
	}
	if (!failed) {
		cout << line_num << " weights have been glued SUCESSFULLY!\n";
	}
}

void fill_input_image(string file_name,
		fms_dt input_image[input_image_depth][input_image_height][input_image_width]) {
	int a;
	std::ifstream infile(file_name);
	bool failed = false;
	int line_num = 0;
	const int input_image_hw = input_image_height * input_image_width;
	while (infile >> a) {
		int channel_index = line_num / input_image_hw;
		int channel_row = (line_num % input_image_hw) / input_image_width;
		int channel_col = line_num % input_image_height;
		line_num++;
		input_image[channel_index][channel_row][channel_col] = (fms_dt) a;
	}
}

void fill_layer_input(string file_name, fms_dt layer_input[max_fms_size],
		const int ofms_h, const int ofms_w) {

	int ofms_hw = ofms_h * ofms_w;
	int num_tiles_h = ofms_h / pw_tile_h;
	int num_tiles_w = ofms_w / pw_tile_w;
	int num_tiles_hw = num_tiles_h * num_tiles_w;

	int a;
	int line = 0;
	std::ifstream infile(file_name);
	while (infile >> a) {
		int z = (line / ofms_hw);
		int h = ((line % ofms_hw) / ofms_w);
		int w = (line % ofms_w);

		int tile_in_z = z / pw_tile_d;
		int tile_in_h = h / pw_tile_h;
		int tile_in_w = w / pw_tile_w;
		int tile_index = tile_in_z * num_tiles_hw + tile_in_h * num_tiles_w
				+ tile_in_w;

		int in_tile_z = z % pw_tile_d;
		int in_tile_h = h % pw_tile_h;
		int in_tile_w = w % pw_tile_w;
		int in_tile_index = in_tile_z * pw_tile_hw + in_tile_h * pw_tile_w
				+ in_tile_w;

		int actual_index = tile_index * pw_tile_size + in_tile_index;

		layer_input[actual_index] = (fms_dt) a;
//		if (line == 12544 + 8) {
//			cout << "\n******tile_in_z*******\n" << tile_in_z
//					<< "\n*************\n";
//			cout << "\n******tile_in_h*******\n" << tile_in_h
//					<< "\n*************\n";
//			cout << "\n******tile_in_w*******\n" << tile_in_w
//					<< "\n*************\n";
//			cout << "\n******tile_index*******\n" << tile_index
//					<< "\n*************\n";
//			cout << "\n******in_tile_z*******\n" << in_tile_z
//					<< "\n*************\n";
//			cout << "\n******in_tile_h*******\n" << in_tile_h
//					<< "\n*************\n";
//			cout << "\n******in_tile_w*******\n" << in_tile_w
//					<< "\n*************\n";
//			cout << "\n******in_tile_index*******\n" << in_tile_index
//					<< "\n*************\n";
//			cout << "\n******actual_index******\n" << actual_index
//					<< "\n*************\n";
//		}
//		if (line == 832 || line == 904 || line == 909
//				|| line ==  888 || line == 896) {
//			cout << "\n" << line << "*****" << actual_index << "\n";
//		}
		line++;
	}
}

void verify_fill_layer_input(string file_name, fms_dt ofms[max_fms_size],
		const int ofms_size, const int ofms_h, const int ofms_w) {

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

		int actual_index = (tile_in_d * pw_tile_d + in_tile_d) * (ofms_h * ofms_w)
				+ (tile_in_h * pw_tile_h + in_tile_h) * ofms_w + tile_in_w * pw_tile_w
				+ in_tile_w;

		to_print_ofms[actual_index] = ofms[i];
//		if (i == 440 || i == 960 || i == 965
//				|| i== 888 || i == 896) {
//			cout << "\n" << i << "#####" << actual_index << "\n";
//		}
//		if (i == 960) {
//			cout << "\n******tile_in_d*******\n" << tile_in_d
//					<< "\n*************\n";
//			cout << "\n******tile_in_h*******\n" << tile_in_h
//					<< "\n*************\n";
//			cout << "\n******tile_in_w*******\n" << tile_in_w
//					<< "\n*************\n";
//
//			cout << "\n******in_tile_d*******\n" << in_tile_d
//					<< "\n*************\n";
//			cout << "\n******in_tile_h*******\n" << in_tile_h
//					<< "\n*************\n";
//			cout << "\n******in_tile_w*******\n" << in_tile_w
//					<< "\n*************\n";
//			cout << "\n******actual_index******\n" << actual_index
//					<< "\n*************\n";
//		}
	}
	myfile.open(file_name);
	for (int i = 0; i < ofms_size; i++) {
		myfile << to_print_ofms[i] << "\n";
	}
	myfile.close();

}
