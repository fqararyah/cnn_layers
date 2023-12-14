

#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main() {
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load("ILSVRC2012_val_00018455.JPEG", &width, &height, &bpp, 3);
    printf("%d \n", width);
    printf("%d \n", height);
    printf("%d \n", bpp);

    stbi_image_free(rgb_image);

    return 0;
}