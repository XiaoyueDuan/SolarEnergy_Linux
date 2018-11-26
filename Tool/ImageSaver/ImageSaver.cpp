//
// Created by dxt on 18-11-26.
//

#include <iomanip>
#include <fstream>
#include <sstream>

#include "ImageSaver.h"
#include "global_constant.h"

void ImageSaver::saveText(std::string filename, int height, int width, float *h_data, int precision, int rows_package) {
    std::ofstream fout(filename.c_str());
    std::stringstream ss;

    int address = 0;
    for (int r = 0; r < height; ++r) {
        if (r % rows_package == rows_package - 1) {
            fout << ss.rdbuf();
            ss.clear();
        }

        for (int c = 0; c < width; ++c) {
            address = (height - 1 - r) * width + c;

            if (h_data[address] < Epsilon) {
                ss << 0;
            } else {
                ss << std::fixed << std::setprecision(precision) << h_data[address];
            }

            if (c != width - 1) {
                ss << ',';
            } else {
                ss << '\n';
            }
        }
    }

    fout << ss.rdbuf();
    fout.close();
}