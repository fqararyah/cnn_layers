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
#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "fibha_host.h"
//#include "prepare_weights_and_inputs.h"
//#include "../../fibha_v2_kernels/src/krnl_fibha_v2.h"

static const std::string error_message = "Error: Result mismatch:\n"
		"i = %d CPU result = %d Device result = %d\n";

#include "../../fiba_v2_kernels/src/all_includes.h"
#include <fstream>
#include "ap_int.h"
#include <iostream>
#include "cstdlib"
#include <chrono>

#include "prepare_weights_and_inputs.h"
#include <iostream>
#include <dirent.h>
#include "test_utils.h"

using namespace std::chrono;

int main(int argc, char* argv[]) {

	//TARGET_DEVICE macro needs to be passed from gcc command line
	if (argc < 2) {
		std::cout << "Usage: " << argv[0]
				<< " <xclbin> number_of_images_to_test" << std::endl;
		return EXIT_FAILURE;
	}

	//*********************************************************************************************************************
	string weights_file = "./off_chip_weights/off_chip_weights.txt";
	string input_images_folder = "./preprocessed_tst_images/";
	string output_folder = "./_FPGA_out/";

	weights_grp_dt glued_weights[all_off_chip_pw_s_weights];
	fms_dt fc_input[fc_layer_input_size];
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel];

	//**********************************************************************************************************************
	std::string xclbinFilename = argv[1];

	std::vector<cl::Device> devices;
	cl::Device device;
	cl_int err;
	cl::Context context;
	cl::CommandQueue q;
	cl::Kernel krnl_fibha_v2;
	cl::Program program;
	std::vector<cl::Platform> platforms;
	bool found_device = false;

	//traversing all Platforms To find Xilinx Platform and targeted
	//Device in Xilinx Platform
	cl::Platform::get(&platforms);
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
			q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

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
	// OpenCL function that is executed on the _FPGA.
	OCL_CHECK(err, krnl_fibha_v2 = cl::Kernel(program, "krnl_fibha_v2", &err));

	// These commands will allocate memory on the Device. The cl::Buffer objects can
	// be used to reference the memory locations on the device.
	OCL_CHECK(err,
			cl::Buffer buffer_weights(context, CL_MEM_READ_ONLY, all_off_chip_pw_s_weights, NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_input_image(context, CL_MEM_READ_ONLY, input_image_depth * input_image_num_fms_groups_in_a_channel*(512/8), NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, fc_layer_input_size, NULL, &err));
	OCL_CHECK(err,
			cl::Buffer buffer_ready_to_receive_a_new_input(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, &err));

	//set the kernel Arguments
	int narg = 0;
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_weights));
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_input_image));
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++, buffer_result));
	OCL_CHECK(err, err = krnl_fibha_v2.setArg(narg++,
					buffer_ready_to_receive_a_new_input));

	int ready_to_receive_a_new_input;

	weights_grp_dt *ptr_weights = glued_weights;
	fms_grp_dt *ptr_input_image = input_image;
	fms_dt *ptr_result = fc_input;
	int *ready_to_receive_a_new_input_ptr = &ready_to_receive_a_new_input;

	OCL_CHECK(err,
			ptr_weights = (weights_grp_dt*)q.enqueueMapBuffer (buffer_weights , CL_TRUE , CL_MAP_WRITE , 0, all_off_chip_pw_s_weights, NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_input_image = (fms_grp_dt*)q.enqueueMapBuffer (buffer_input_image , CL_TRUE , CL_MAP_WRITE , 0, input_image_depth * input_image_num_fms_groups_in_a_channel*(512/8), NULL, NULL, &err));
	OCL_CHECK(err,
			ptr_result = (fms_dt*)q.enqueueMapBuffer (buffer_result , CL_TRUE , CL_MAP_READ , 0, fc_layer_input_size, NULL, NULL, &err));
	OCL_CHECK(err,
			ready_to_receive_a_new_input_ptr = (int*)q.enqueueMapBuffer (buffer_ready_to_receive_a_new_input , CL_TRUE , CL_MAP_READ , 0, sizeof(int), NULL, NULL, &err));
	//*********************************************************************************************************************8

	DIR *dir;
	int img_count = 0;
	int images_to_test = 1000;
	if (argc == 3) {
		images_to_test = atoi(argv[2]);
	}
	struct dirent *ent;
	// Data will be migrated to kernel space
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects( { buffer_weights}, 0/* 0 means from host*/));
	glue_weights(weights_file, ptr_weights);
	if ((dir = opendir(input_images_folder.c_str())) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			string file_name = input_images_folder + ent->d_name;
			if (file_name.find(".txt") == std::string::npos) {
				continue;
			}
			cout << file_name << "\n";
			glue_input_image(file_name, ptr_input_image);
			OCL_CHECK(err, err = q.enqueueMigrateMemObjects( { buffer_input_image}, 0/* 0 means from host*/));
			//verify_input_image(input_image_v_file, input_image);TTTTTTTTTTTTTTTTTTTTTTTTTTTT

			OCL_CHECK(err, q.finish());
			//Launch the Kernel
			OCL_CHECK(err, err = q.enqueueTask(krnl_fibha_v2));

			auto start = high_resolution_clock::now();
						OCL_CHECK(err, q.finish());
						auto stop = high_resolution_clock::now();
						auto duration = duration_cast<microseconds>(stop - start);
						cout << duration.count() << endl;

			// The result of the previous kernel execution will need to be retrieved in
			// order to view the results. This call will transfer the data from _FPGA to
			// source_results vector
			OCL_CHECK(err,
					q.enqueueMigrateMemObjects({buffer_result},CL_MIGRATE_MEM_OBJECT_HOST));

			OCL_CHECK(err, q.finish());

//			while (*ready_to_receive_a_new_input_ptr != 1) {
//				std::cout << "not ready yet!\n";
//			}

			//The result
			dump_ouput(output_folder + ent->d_name, ptr_result,
					fc_layer_input_size);
			img_count++;

			std::cout << "readiness: " << *ready_to_receive_a_new_input_ptr
					<< "\n";
			//*******************************************************************************************************************
			if (img_count == images_to_test) {
				break;
			}
		}
		closedir(dir);
	} else {
		return EXIT_FAILURE;
	}
	cout << img_count;

	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_weights, ptr_weights));
	OCL_CHECK(err,
			err = q.enqueueUnmapMemObject(buffer_input_image, ptr_input_image));
	OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_result, ptr_result));
	OCL_CHECK(err,
			err = q.enqueueUnmapMemObject(buffer_ready_to_receive_a_new_input,
					ready_to_receive_a_new_input_ptr));
	OCL_CHECK(err, err = q.finish());

	//std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;

	return (EXIT_SUCCESS);

}
