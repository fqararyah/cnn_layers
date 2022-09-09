

void pw_conv(weights_grp_dt *weights, fms_dt channels[max_fms_size],
		fms_dt result[max_fms_size], int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w, fms_dt tmp_channels[max_tmp_fms_size],
		int read_write, const int num_of_weight_groups,
		const normalization_scheme normalization, const int direction, const int layer_weights_offset);