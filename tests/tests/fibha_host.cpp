/*******************************************************************************
 Vendor: Xilinx
 Associated Filename: vadd.cpp
 Purpose: VITIS vector addition

 *******************************************************************************
 Copyright (C) 2019 XILINX, Inc.

 This file contains confidential and proprietary information of Xilinx, Inc. and
 is protected under U.S. and international copyright and other intellectual
 property laws.

 DISCLAIMER
 This disclaimer is not a license and does not grant any rights to the materials
 distributed herewith. Except as otherwise provided in a valid license issued to
 you by Xilinx, and to the maximum extent permitted by applicable law:
 (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX
 HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
 INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
 FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
 in contract or tort, including negligence, or under any other theory of
 liability) for any loss or damage of any kind or nature related to, arising under
 or in connection with these materials, including for any direct, or any indirect,
 special, incidental, or consequential loss or damage (including loss of data,
 profits, goodwill, or any type of loss or damage suffered as a result of any
 action brought by a third party) even if such damage or loss was reasonably
 foreseeable or Xilinx had been advised of the possibility of the same.

 CRITICAL APPLICATIONS
 Xilinx products are not designed or intended to be fail-safe, or for use in any
 application requiring fail-safe performance, such as life-support or safety
 devices or systems, Class III medical devices, nuclear facilities, applications
 related to the deployment of airbags, or any other applications that could lead
 to death, personal injury, or severe property or environmental damage
 (individually and collectively, "Critical Applications"). Customer assumes the
 sole risk and liability of any use of Xilinx products in Critical Applications,
 subject only to applicable laws and regulations governing limitations on product
 liability.

 THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
 ALL TIMES.

 *******************************************************************************/
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "cstdlib"
#include <chrono>
#include <dirent.h>

#include "fibha_host.h"
#include "../../fiba_v2_kernels/src/model_components/model/headers/mob_v2_layers_specs.h"
#include "client/fpga_cpp_fc.h"
#include "tests/fpga_test_utils.h"
#include "client/fpga_prepare_weights_and_inputs.h"

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

static const std::string error_message = "Error: Result mismatch:\n"
		"i = %d CPU result = %d Device result = %d\n";

#define PROFILING 1

using namespace std::chrono;

int main(int argc, char* argv[]) {

	//TARGET_DEVICE macro needs to be passed from gcc command line
	if (argc < 2) {
		std::cout << "Usage: " << argv[0]
				<< " <xclbin> number_of_images_to_test" << std::endl;
		return EXIT_FAILURE;
	}

	//*********************************************************************************************************************
	string weights_file = "/media/sd-mmcblk0p2/off_chip_weights/"
			+ get_model_prefix() + "_off_chip_weights_fpga.txt";

	string dw_weights_file = "/media/sd-mmcblk0p2/off_chip_weights/"
			+ get_model_prefix() + "_off_chip_dw_weights_pipeline_"
			+ to_string(PIPELINE_LENGTH) + ".txt";

	string on_chip_weights_file =
			"/media/sd-mmcblk0p2/on_chip_weights/" + get_model_prefix() + "_on_chip_weights.txt";

	string fused_scales_file = "/media/sd-mmcblk0p2/off_chip_weights/"
			+ get_model_prefix() + "_fused_scales_pipeline_"
			+ to_string(PIPELINE_LENGTH) + ".txt";
	string fused_zps_file = "/media/sd-mmcblk0p2/off_chip_weights/"
			+ get_model_prefix() + "_fused_zps_pipeline_"
			+ to_string(PIPELINE_LENGTH) + ".txt";

	string fc_weights_file = "/media/sd-mmcblk0p2/off_chip_weights/"
			+ get_model_prefix() + "_fc_weights.txt";
	string weight_sums_file = "/media/sd-mmcblk0p2/off_chip_weights/"
			+ get_model_prefix() + "_fc_weight_sums.txt";
	string biases_file = "/media/sd-mmcblk0p2/off_chip_weights/"
			+ get_model_prefix() + "_fc_biases.txt";

	string input_images_folder = "/media/sd-mmcblk0p2/resized_images/";
	string output_folder = "/media/sd-mmcblk0p2/fpga_out/";

	string predictions_file =
			"/media/sd-mmcblk0p2/fpga_out/predictions_fpga.json";

	//weights_grp_dt glued_weights[all_pw_weights];
	//fms_dt fc_input[fc_layer_input_size];
	//fms_grp_dt input_image[input_image_depth
//			* input_image_num_fms_groups_in_a_channel];

	int8_t fc_weights[fc_cols * fc_layer_input_size];
	int64_t weight_sums[num_classes];
	int biases[num_classes];
	string predictions_file_content = "[";
	int top5[5];

	//**********************************************************************************************************************
	std::string xclbinFilename = argv[1];

	std::vector < cl::Device > devices;
	cl::Device device;
	cl_int err;
	cl::Context context;
	cl::CommandQueue q;
	cl::Kernel krnl_fibha_v2;
	cl::Program program;
	std::vector < cl::Platform > platforms;
	bool found_device = false;

	//traversing all Platforms To find Xilinx Platform and targeted
	//Device in Xilinx Platform
	cl::Platform::get (&platforms);
	for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++) {
		cl::Platform platform = platforms[i];
		std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
		if (platformName == "Xilinx") {
			devices.clear();
			platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
			if (devices.size()) {
				device = devices[0];
				found_device = true;
				break;
			}
		}
	}
	if (found_device == false) {
		std::cout << "Error: Unable to find Target Device "
				<< device.getInfo<CL_DEVICE_NAME>() << std::endl;
		return EXIT_FAILURE;
	}

	// Creating Context and Command Queue for selected device
	OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
	OCL_CHECK(err,
			q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE,
					&err));

	std::cout << "INFO: Reading " << xclbinFilename << std::endl;
	FILE* fp;
	if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
		printf("ERROR: %s xclbin not available please build\n",
				xclbinFilename.c_str());
		exit(EXIT_FAILURE);
	}
	// Load xclbin
	std::cout << "Loading: '" << xclbinFilename << "'\n";
	std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
	bin_file.seekg(0, bin_file.end);
	unsigned nb = bin_file.tellg();
	bin_file.seekg(0, bin_file.beg);
	char *buf = new char[nb];
	bin_file.read(buf, nb);

	// Creating Program from Binary File
	cl::Program::Binaries bins;
	bins.push_back( { buf, nb });
	devices.resize(1);
	OCL_CHECK(err, program = cl::Program(context, devices, bins, NULL, &err));

	// This call will get the kernel object from program. A kernel is an
	// OpenCL function that is executed on the FPGA.
	OCL_CHECK(err, krnl_fibha_v2 = cl::Kernel(program, "krnl_fibha_v2", &err));

	// These commands will allocate memory on the Device. The cl::Buffer objects can
	// be used to reference the memory locations on the device.
	OCL_CHECK(err,
			cl::Buffer buffer_input_image(context, CL_MEM_READ_ONLY, input_image_depth * input_image_num_fms_groups_in_a_channel*sizeof(fms_grp_dt), NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_weights(context, CL_MEM_READ_ONLY, all_off_chip_pw_s_weights * sizeof(weights_grp_dt), NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_dw_weights(context, CL_MEM_READ_ONLY, all_dw_off_chip_weights * sizeof(weights_dt), NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_off_chip_fused_scales(context, CL_MEM_READ_ONLY, all_off_chip_fused_scales_zps * sizeof(fused_scales_dt), NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_off_chip_fused_zeropoints(context, CL_MEM_READ_ONLY, all_off_chip_fused_scales_zps * sizeof(biases_dt), NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_on_chip_weights(context, CL_MEM_READ_ONLY, all_on_chip_pw_s_weights_groups * sizeof(weights_grp_dt), NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, fc_layer_input_size * sizeof(fms_dt), NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_model_config_list(context, CL_MEM_WRITE_ONLY, 2 * max_conv_layers * sizeof(int), NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_first_lunch(context, CL_MEM_WRITE_ONLY, sizeof(int*), NULL, &err));

	//set the kernel Arguments
	int narg = 0;
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_input_image));
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_weights));
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_dw_weights));
	OCL_CHECK(err,
			err = krnl_fibha_v2.setArg(narg++, buffer_off_chip_fused_scales));
	OCL_CHECK(err,
			err = krnl_fibha_v2.setArg(narg++,
					buffer_off_chip_fused_zeropoints));
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_on_chip_weights));
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_result));
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_model_config_list));
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_first_lunch));

	weights_grp_dt *ptr_weights;
	weights_dt *ptr_dw_weights;
	fused_scales_dt *ptr_off_chip_fused_scales;
	biases_dt *ptr_off_chip_fused_zeropoints;
	weights_grp_dt *ptr_glued_on_chip_weights;// = (weights_grp_dt *)malloc(all_on_chip_pw_s_weights_groups *
											  //							 weights_group_items * weights_dt_width / 8);
	fms_grp_dt *ptr_input_image;
	fms_dt *ptr_result;
	int *ptr_model_config_list;
	int *ptr_first_lunch;

	OCL_CHECK(err,
				ptr_input_image = (fms_grp_dt*)q.enqueueMapBuffer (buffer_input_image , CL_TRUE , CL_MAP_WRITE , 0, input_image_depth * input_image_num_fms_groups_in_a_channel*sizeof(fms_grp_dt), NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_weights = (weights_grp_dt*)q.enqueueMapBuffer (buffer_weights , CL_TRUE , CL_MAP_WRITE , 0, all_off_chip_pw_s_weights * sizeof(weights_grp_dt), NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_dw_weights = (weights_dt*)q.enqueueMapBuffer (buffer_dw_weights , CL_TRUE , CL_MAP_WRITE , 0, all_dw_off_chip_weights *sizeof(weights_dt), NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_off_chip_fused_scales = (fused_scales_dt*)q.enqueueMapBuffer (buffer_off_chip_fused_scales , CL_TRUE , CL_MAP_WRITE , 0, all_off_chip_fused_scales_zps*sizeof(fused_scales_dt), NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_off_chip_fused_zeropoints = (biases_dt*)q.enqueueMapBuffer (buffer_off_chip_fused_zeropoints , CL_TRUE , CL_MAP_WRITE , 0, all_off_chip_fused_scales_zps*sizeof(biases_dt), NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_glued_on_chip_weights = (weights_grp_dt*)q.enqueueMapBuffer (buffer_on_chip_weights , CL_TRUE , CL_MAP_WRITE , 0, all_on_chip_pw_s_weights_groups*sizeof(weights_grp_dt), NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_result = (fms_dt*)q.enqueueMapBuffer (buffer_result , CL_TRUE , CL_MAP_READ , 0, fc_layer_input_size * sizeof(fms_dt), NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_model_config_list = (int*)q.enqueueMapBuffer (buffer_model_config_list , CL_TRUE , CL_MAP_READ , 0, 2 * max_conv_layers * sizeof(int), NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_first_lunch = (int*)q.enqueueMapBuffer (buffer_first_lunch , CL_TRUE , CL_MAP_READ , 0, sizeof(int*), NULL, NULL, &err));
	//*********************************************************************************************************************8

	DIR *dir;
	int img_count = 0;
	int images_to_test = 1000;
	if (argc >= 3) {
		images_to_test = atoi(argv[2]);
	}
	if (argc >= 4) {
		//read_model_configs(argv[3], ptr_model_config_list);
		cout << "model_configs_list is filled *************\n";
	} else {
		for (int i = 0; i < 2 * max_conv_layers - 1; i++) {
			ptr_model_config_list[i] = 0;
		}
	}

	*ptr_first_lunch = 1;

	struct dirent *ent;
//	 Data will be migrated to kernel space
	load_weights(dw_weights_file, ptr_dw_weights);
	load_fused_scales(fused_scales_file, ptr_off_chip_fused_scales);
	load_fused_zps(fused_zps_file, ptr_off_chip_fused_zeropoints);
	glue_weights(weights_file, ptr_weights);
	validate_weights(weights_file, ptr_weights);
	glue_on_chip_weights_fpga(on_chip_weights_file, ptr_glued_on_chip_weights);
	read_fc_weights(fc_weights_file, fc_weights);
	read_weight_sums(weight_sums_file, weight_sums);
	read_biases(biases_file, biases);

	OCL_CHECK(err,
			err = q.enqueueMigrateMemObjects( { buffer_weights },
					0/* 0 means from host*/));

	OCL_CHECK(err,
			err = q.enqueueMigrateMemObjects( { buffer_dw_weights },
					0/* 0 means from host*/));

	OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {
			buffer_off_chip_fused_scales }, 0/* 0 means from host*/));

	OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {
			buffer_off_chip_fused_zeropoints }, 0/* 0 means from host*/));

	OCL_CHECK(err,
			err = q.enqueueMigrateMemObjects( { buffer_on_chip_weights },
					0/* 0 means from host*/));

	OCL_CHECK(err,
			err = q.enqueueMigrateMemObjects( { buffer_model_config_list },
					0/* 0 means from host*/));

	OCL_CHECK(err,
			err = q.enqueueMigrateMemObjects( { buffer_first_lunch },
					0/* 0 means from host*/));

	if ((dir = opendir(input_images_folder.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			string file_name = input_images_folder + ent->d_name;
			if (file_name.find(".JPEG") == std::string::npos) {
				continue;
			}
			string formatted_file_name = ((string) ent->d_name).substr(0,
					((string) ent->d_name).find(".", 0) + 1) + "JPEG";
			cout << file_name << "\n";
			glue_and_quantize_input_image(file_name, ptr_input_image, quantize_layer_specs);

//			verify_glued_image(file_name, ptr_input_image);
//			std::cout<<(fms_dt)ptr_weights[0](fms_dt_offset,0)<<" "
//											<<(fms_dt)ptr_weights[1](fms_dt_offset,0)<<" "
//											<<(fms_dt)ptr_weights[2](fms_dt_offset,0)
//											<<" "<<(fms_dt)ptr_weights[10000](fms_dt_offset,0)<<"\n";
			OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {
					buffer_input_image }, 0/* 0 means from host*/));
			OCL_CHECK(err, q.finish());
#if PROFILING
			auto start = high_resolution_clock::now();
#endif
			//Launch the Kernel
			OCL_CHECK(err, err = q.enqueueTask(krnl_fibha_v2));
			OCL_CHECK(err, q.finish());
#if PROFILING
			auto stop = high_resolution_clock::now();
			auto duration = duration_cast<microseconds>(stop - start);
			cout << duration.count() << endl;
#endif
			// The result of the previous kernel execution will need to be retrieved in
			// order to view the results. This call will transfer the data from FPGA to
			// source_results vector
			OCL_CHECK(err,
					q.enqueueMigrateMemObjects( { buffer_result },
							CL_MIGRATE_MEM_OBJECT_HOST));
			OCL_CHECK(err, q.finish());

//			for (int i = 0; i < 2 * max_conv_layers; i++) {
//				cout << ptr_result[i] << " " << ptr_result[i + 500] << "\n";
//			}
			if ((*ptr_first_lunch) == 1) {
				*ptr_first_lunch = 0;
				OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {
						buffer_first_lunch }, 0/* 0 means from host*/));
			}

//			std::cout << ptr_result[0] << " " << ptr_result[1] << " "
//					<< ptr_result[2] << " " << ptr_result[3] << " "
//					<< ptr_result[4] << " " << ptr_result[5] << " "
//					<< ptr_result[500] << " " << ptr_result[501] << " "
//					<< ptr_result[502] << " " << ptr_result[503] << " ";
#if PROFILING
			start = high_resolution_clock::now();
			cout << duration.count() << endl;
#endif
			fc_layer(ptr_result, fc_weights, weight_sums, top5, biases,
					layer_68_fc_specs);
#if PROFILING
			stop = high_resolution_clock::now();
			duration = duration_cast<microseconds>(stop - start);
			cout << "fc_duration: " << duration.count() << endl;
#endif
//			while (*ready_to_receive_a_new_input_ptr != 1) {
//				std::cout << "not ready yet!\n";
//			}

			//The result
//			dump_ouput(output_folder + ent->d_name, ptr_result,
//					fc_layer_input_size);
			predictions_file_content += top_5_to_predictions_dict(top5,
					formatted_file_name);
			img_count++;

			//*******************************************************************************************************************
			if (img_count == images_to_test) {
				break;
			}
		}
		closedir(dir);
	} else {
		return EXIT_FAILURE;
	}

	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_weights, ptr_weights));
	OCL_CHECK(err,
			err = q.enqueueUnmapMemObject(buffer_dw_weights, ptr_dw_weights));
	OCL_CHECK(err,
			err = q.enqueueUnmapMemObject(buffer_off_chip_fused_scales,
					ptr_off_chip_fused_scales));
	OCL_CHECK(err,
			err = q.enqueueUnmapMemObject(buffer_off_chip_fused_zeropoints,
					ptr_off_chip_fused_zeropoints));
	OCL_CHECK(err,
			err = q.enqueueUnmapMemObject(buffer_on_chip_weights,
					ptr_glued_on_chip_weights));
	OCL_CHECK(err,
			err = q.enqueueUnmapMemObject(buffer_input_image, ptr_input_image));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_result, ptr_result));
	OCL_CHECK(err,
			err = q.enqueueUnmapMemObject(buffer_model_config_list,
					ptr_model_config_list));
	OCL_CHECK(err,
			err = q.enqueueUnmapMemObject(buffer_first_lunch, ptr_first_lunch));
	OCL_CHECK(err, err = q.finish());

	//std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;

	cout << img_count;
	predictions_file_content = predictions_file_content.substr(0,
			predictions_file_content.length() - 1) + ']';
	save_predictions(predictions_file, predictions_file_content);

	return (EXIT_SUCCESS);

}
