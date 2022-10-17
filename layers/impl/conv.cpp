#include "../headers/conv.h"
#include "../headers/pw_conv.h"
#include "../headers/norm_act.h"
#include "../../utils/utils.h"
#include "../../client/quantization_and_biases.h"
#include "../../tests/test_utils.h"

void fill_channels_buffer_0(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt channels_tile[pw_tile_d][pw_tile_h][pw_tile_w], int starting_d,
		int starting_h, int starting_w) {
#pragma HLS INLINE
	for (int t_d = 0; t_d < pw_tile_d; t_d++) {
		for (int t_h = 0; t_h < pw_tile_h; t_h++) {
			for (int t_w = 0; t_w < pw_tile_w; t_w++) {
				channels_tile[t_d][t_h][t_w] =
						channels[starting_d + t_d][starting_h
								+ t_h * layer_0_strides][starting_w
								+ t_w * layer_0_strides];
			}
		}
	}
}

void write_results_tile_0(
		pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		fms_dt results[max_fms_size], int tile_indx, const int layer_conv_d,
		int starting_d, int layer, const int num_of_tiles_hw) {

	biases_dt fused_zero_points_buffer[pw_conv_parallelism_out];
	scales_dt fused_scales_buffer[pw_conv_parallelism_out];
	fill_fused_zero_points(fused_zero_points, fused_zero_points_buffer,
			starting_d, layer);
	fill_fused_scales(fused_scales, fused_scales_buffer, starting_d, layer);

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };

	for (int tile_offset = 0; tile_offset < pw_conv_parallelism_out / pw_tile_d;
			tile_offset++) {
#pragma HLS PIPELINE
//#pragma HLS dependence variable = results inter false
//#pragma HLS dependence variable = results intra false
		const int current_fms_indx = (tile_indx + tile_offset * num_of_tiles_hw)
				* pw_tile_size;
		//cout << current_fms_indx << "\n";
		for (int t_h = 0; t_h < pw_tile_h; t_h++) {
#pragma HLS UNROLL
			for (int t_w = 0; t_w < pw_tile_w; t_w++) {
#pragma HLS UNROLL
				for (int t_d = 0; t_d < pw_tile_d; t_d++) {
#pragma HLS UNROLL
					if (t_d < layer_conv_d) {
						const int in_tile_index = tile_offset * pw_tile_d + t_d;
						normalization.fused_zero_point =
								fused_zero_points_buffer[in_tile_index];
						normalization.fused_scales =
								fused_scales_buffer[in_tile_index];
						normalization.ofm_zero_point =
								conv_fms_zero_points[layer + 2];
						normalization.ofm_scale = conv_fms_scales[layer + 2];
						fms_dt scaled_val =
								conv_relu_norm(
										results_tile[tile_offset * pw_tile_d
												+ t_d][t_h][t_w], normalization,
										6);
						if (current_fms_indx == 0) {
							pss_f_dt scaled_pss =
									(pss_f_dt) (normalization.fused_scales
											* (results_tile[tile_offset
													* pw_tile_d + t_d][t_h][t_w]
													+ normalization.fused_zero_point));
//							cout << normalization.fused_scales << "*" << "("
//									<< results_tile[tile_offset * pw_tile_d
//											+ t_d][t_h][t_w] << "+"
//									<< normalization.fused_zero_point << ") = "
//									<< scaled_pss<<"\n";
//							if(t_d == 0){
//								for(int i=0;i<4;i++){
//									cout<<fused_scales[i]<<", ";
//								}
//								cout<<"\n";
							}
//							cout
//									<<in_tile_index <<": "<< results_tile[tile_offset * pw_tile_d
//											+ t_d][t_h][t_w] << " > "
//									<< scaled_val << " using " << normalization.fused_zero_point <<" "<<
//									normalization.fused_scales << " " << normalization.ofm_zero_point << "\n";
						}
						results[current_fms_indx + t_d * pw_tile_hw
								+ t_h * pw_tile_w + t_w] = scaled_val;
					}
				}
			}
		}
	}
}

//Note that this implementation of layer_0 is not efficient, it is just the easiest to use pw_con_eng
void layer_0_using_pw(
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3],
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size], const int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w) {
#pragma HLS INLINE off

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };
	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d];

#pragma HLS ARRAY_PARTITION variable = weights_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights_tile cyclic factor = pw_conv_parallelism_in dim = 2

	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;
	const int step_h = layer_0_strides * pw_tile_h;
	const int step_w = layer_0_strides * pw_tile_w;
	const int step_d = pw_tile_d;
	const int num_of_tiles_d = layer_0_depth / pw_tile_d;

	conv2_ots_loop: for (int td_o = 0; td_o < num_of_tiles_d_out; td_o++) {
		conv2_ith_loop: for (int t_in_h = 0; t_in_h < num_of_tiles_h;
				t_in_h++) {
			//############width loop##############
			conv2_itw_loop: for (int t_in_w = 0; t_in_w < num_of_tiles_w;
					t_in_w++) {
				//############width loop##############
				pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w] =
						{ 0 };
#pragma HLS ARRAY_PARTITION variable = results_tile complete dim = 0

				for (int c_h = 0; c_h < 3; c_h++) {
					for (int c_w = 0; c_w < 3; c_w++) {

						//fill filters
						for (int fil = 0; fil < pw_conv_parallelism_out;
								fil++) {
							for (int d = 0; d < 3; d++) {
								weights_tile[fil][d] =
										(weights_dt) weights_0[fil
												+ td_o * pw_conv_parallelism_out][d][c_h][c_w];
							}
						}
						//end fill filters

						conv2_itd_loop: for (int t_in_d = 0;
								t_in_d < num_of_tiles_d; t_in_d++) {
							//############depth loop##############
							fms_dt channels_buffer[pw_tile_d][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = channels_buffer complete dim = 0

//							if (c_h == 0 && c_w == 0 && t_in_d * step_d == 1
//									&& t_in_h * step_h == 0
//									&& t_in_w * step_w == 0 && td_o == 0) {
//								cout << "1 " << results_tile[0][0][0] << "\n";
//							}
							fill_channels_buffer_0(channels, channels_buffer,
									t_in_d * step_d, t_in_h * step_h + c_h,
									t_in_w * step_w + c_w);
//							if (c_h == 0 && c_w == 0 && t_in_d * step_d == 1
//									&& t_in_h * step_h == 0
//									&& t_in_w * step_w == 0 && td_o == 0) {
//								cout << "2 " << results_tile[0][0][0] << "\n";
//							}
							pw_conv_eng(channels_buffer, weights_tile,
									results_tile, t_in_d * step_d,
									td_o * pw_conv_parallelism_out,
									layer_conv_d, layer_num_fils);
//							if (c_h == 0 && c_w == 0 && t_in_d * step_d == 1
//									&& t_in_h * step_h == 0
//									&& t_in_w * step_w == 0 && td_o == 0) {
//								cout << "3 " << results_tile[0][0][0]
//										<< "\n";
////								cout << "3 " << channels[0][0][0] << "\n";
////								cout << "3 " << channels[1][0][0] << "\n";
////								cout << "3 " << channels[2][0][0] << "\n";
//							}
//							//if(c_h == 0 && c_w == 0 && t_in_d * step_d == 0 && t_in_h * step_h == 0 && t_in_w * step_w == 0){cout<<results_tile[0][0][0]<<"\n";}
//							//cout<<t_in_d * step_d <<" "<< t_in_h * step_h + c_h <<" "<< step_w + c_w<<"\n";
//							if (t_in_h * step_h == 0
//									&& t_in_w * step_w == 0 && td_o == 0) {
//								dumb_pw_channels_tile(
//										"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tile_ch"+to_string(t_in_d)+"_"
//												+ to_string(c_h * 3 + c_w)
//												+ ".txt", channels_buffer);
//								dumb_pw_weights_tile(
//										"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tile_w"+to_string(t_in_d)+"_"
//												+ to_string(c_h * 3 + c_w)
//												+ ".txt", weights_tile,
//										layer_conv_d);
//								dumb_pw_pss_tile(
//										"/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/tile_pss"+to_string(t_in_d)+"_"
//												+ to_string(c_h * 3 + c_w)
//												+ ".txt", results_tile);
//							}
						}
					}
				}
				write_results_tile_0(results_tile, result,
						td_o * (pw_conv_parallelism_out / pw_tile_d)
								* num_of_tiles_hw + t_in_h * num_of_tiles_w
								+ t_in_w, layer_num_fils,
						td_o * pw_conv_parallelism_out, layer, num_of_tiles_hw);
			}
		}
	}
}
