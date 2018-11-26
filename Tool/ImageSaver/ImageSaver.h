//
// Created by dxt on 18-11-26.
//

#ifndef SOLARENERGYRAYTRACING_IMAGESAVER_H
#define SOLARENERGYRAYTRACING_IMAGESAVER_H

#include <string>

class ImageSaver {
public:
    static void
    saveText(std::string filename, int height, int width, float *h_data, int precision = 2, int rows_package = 10);
};


#endif //SOLARENERGYRAYTRACING_IMAGESAVER_H
