#include "../client/seml.hpp"
#include "prepare_weights.hpp"
#include <iostream>
using namespace std;

int main(){
	string weights_file = "/media/SSD2TB/wd/cnn_layers/off_chip_weights/off_chip_weights.txt";
	weights_grp_dt glued_weights[max_fms_size / weights_group_items];
	glue_weights(weights_file, glued_weights);
	validate_weights(weights_file, glued_weights);
	return 0;
}
