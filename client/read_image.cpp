#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

void read_image(string file_name,
                char image[])
{
    char a;
    std::ifstream infile(file_name);
    assert(!infile.fail());
    bool failed = false;
    int line_num = 0;
    while (infile >> a)
    {
        if (line_num >= 2 * 224 * 224)
        {
            printf("XXXXXXXXXXXXXXXXXXXx\n");
            break;
        }
        image[line_num] = a;
        line_num++;
    }
}

int main()
{
    string image_file =
        "/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012/ILSVRC2012_val_00018455.JPEG";
    char image[4 * 224 * 224];
    read_image(
        image_file, image);
    for (int i = 0; i < 20; i++)
    {
        for (int j = 0; j < 20; j++)
        {
            printf("%d ", image[i * 20 + j]);
        }
        printf("\n");
    }
    return 0;
}