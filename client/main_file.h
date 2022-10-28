#ifndef MAIN_FILE
#define MAIN_FILE

#include "seml.h"

void top_func(fms_dt input_image[input_image_depth][input_image_height][input_image_width],
		weights_grp_dt off_chip_weights[all_pw_weights], fms_dt fc_input[fc_layer_input_size]);

#endif
