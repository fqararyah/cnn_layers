#include <cstdint>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <chrono>
#include <cassert>

using namespace std;

#ifndef CPP_FC
#define CPP_FC
const int num_classes = 1000;

const double weights_scale = 1.873968634754419327e-03;
const double ifms_scale = 0.020379824563860893;
const double biases_scale = 3.819115227088332176e-05;
const int64_t ifm_zero_point = -128;

void read_ifms(string file_name,
               int8_t ifms[]);

void read_fc_weights(string file_name,
                  int8_t fc_weights[]);

void read_weight_sums(string file_name,
                      int64_t fc_weight_sums[]);

void read_biases(string file_name,
                 int fc_biases[]);

void save_predictions(string file_name, string predictions);

string top_5_to_predictions_dict(int top5[5], string image_name);

void fc_layer(fms_dt in_vector[], int8_t weights[], int64_t weight_sums[], int top5[5], int biases[]);

#endif
