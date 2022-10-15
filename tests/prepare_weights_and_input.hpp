#ifndef PREPARE_WEIGHTS_INPUTS
#define PREPARE_WEIGHTS_INPUTS

#include "../client/seml.h"
#include <fstream>
#include "ap_int.h"
#include <iostream>
using namespace std;

bool isNumber(string& str)
{
    for (char const &c : str) {
        if (std::isdigit(c) == 0)
          return false;
    }
    return true;
}

void glue_weights(string file_name, weights_grp_dt glued_weights[max_fms_size / weights_group_items]) {
	int a;
	std::ifstream infile(file_name);

	int line_num = 0;
	while (infile >> a) {
		weights_dt weight = (weights_dt) a;
		int external_index = line_num / weights_group_items;
		int internal_index  = line_num % weights_group_items;
		glued_weights[external_index](internal_index*weights_dt_width + weights_dt_offset, internal_index*weights_dt_width) = weight;
		line_num++;
	}
}

void validate_weights(string file_name, weights_grp_dt glued_weights[max_fms_size / weights_group_items]) {
	int a;
	std::ifstream infile(file_name);
	bool failed = false;
	int line_num = 0;
	while (infile >> a) {
		weights_dt weight = (weights_dt) a;
		int external_index = line_num / weights_group_items;
		int internal_index  = line_num % weights_group_items;
		if(weight != (weights_dt) glued_weights[external_index](internal_index*weights_dt_width + weights_dt_offset, internal_index*weights_dt_width)){
			cout<<"failed at: "<< line_num << " " << weight << " != "<<
					(weights_dt) glued_weights[external_index](internal_index*weights_dt_width + weights_dt_offset, internal_index*weights_dt_width);
			failed = true;
			break;
		}
		line_num++;
	}
	if(!failed){
		cout<<line_num<<" weights have been glued SUCESSFULLY!\n";
	}
}

void fill_input_image(string file_name, fms_dt input_image[input_image_depth][input_image_height][input_image_width]){
	int a;
	std::ifstream infile(file_name);
	bool failed = false;
	int line_num = 0;
	const int input_image_hw = input_image_height * input_image_width;
	while (infile >> a) {
		int channel_index = line_num/input_image_hw;
		int channel_row = (line_num % input_image_hw) / input_image_width;
		int channel_col = line_num % input_image_height;
		input_image[channel_index][channel_row][channel_col] = (fms_dt)a;
	}
}

#endif
