#include "../client/seml.hpp"
#include <fstream>
#include <ap_utils.h>

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
	std::ifstream infile("thefile.txt");

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
	std::ifstream infile("thefile.txt");
	bool failed = false;
	int line_num = 0;
	while (infile >> a) {
		weights_dt weight = (weights_dt) a;
		int external_index = line_num / weights_group_items;
		int internal_index  = line_num % weights_group_items;
		if(weight != glued_weights[external_index](internal_index*weights_dt_width + weights_dt_offset, internal_index*weights_dt_width)){
			cout<<"failed at: "<< line_num << " " << weight << " != "<<
					glued_weights[external_index](internal_index*weights_dt_width + weights_dt_offset, internal_index*weights_dt_width);
			failed = true;
			break;
		}
		line_num++;
	}
	if(!failed){
		cout<<"SUCESS!\n";
	}
}

