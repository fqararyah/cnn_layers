

#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load("/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012_resized/ILSVRC2012_val_00018455.JPEG", &width, &height, &bpp, 3);
    printf("%d \n", width);
    printf("%d \n", height);
    printf("%d \n", bpp);

    printf("%d ", (int) rgb_image[0]);
    printf("%d ", (int) rgb_image[1]);
    printf("%d ", (int) rgb_image[2]);
    printf("%d ", (int) rgb_image[224]);
    printf("%d ", (int) rgb_image[224+1]);
    printf("%d ", (int) rgb_image[224+2]);
    printf("%d ", (int) rgb_image[2*224*224]);
    printf("%d ", (int) rgb_image[2*224*224+1]);
    printf("%d ", (int) rgb_image[2*224*224+2]);

    stbi_image_free(rgb_image);

    return 0;
}