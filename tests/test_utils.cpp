#include "test_utils.h"

void fill_layer_input_from_file(string file_name, int input_size){

}

void dumb_layer_output(string file_name, fms_dt ofms[max_fms_size], int ofms_size){
    ofstream myfile;
    myfile.open (file_name);
    for(int i=0;i<ofms_size;i++){
        myfile << ofms[i]<<"\n";
    }
    myfile.close();
}
