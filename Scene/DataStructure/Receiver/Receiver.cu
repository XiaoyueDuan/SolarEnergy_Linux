#include "Receiver.cuh"
#include "check_cuda.h"
#include "global_function.cuh"

Receiver::~Receiver() {
    if (d_image_)
        d_image_ = nullptr;
}

void Receiver::CClear() {
    if (d_image_) {
        checkCudaErrors(cudaFree(d_image_));
        d_image_ = nullptr;
    }
}

void Receiver::Calloc_image() {
    checkCudaErrors(cudaMalloc((void **) &d_image_, sizeof(float) * resolution_.x * resolution_.y));
}

void Receiver::Cclean_image_content() {
    int n_resolution = resolution_.x * resolution_.y;
    float *h_clean_receiver = new float[n_resolution];
    for (int i = 0; i < n_resolution; ++i) {
        h_clean_receiver[i] = 0.0f;
    }

    // clean screen
    global_func::cpu2gpu(d_image_, h_clean_receiver, n_resolution);

    delete[] h_clean_receiver;
    h_clean_receiver = nullptr;
}

/**
 * Getter and Setters of attributes for Receiver
 */

int Receiver::getType() const {
    return type_;
}

void Receiver::setType(int type) {
    type_ = type;
}

float3 Receiver::getNormal() const {
    return normal_;
}

void Receiver::setNormal(float3 normal) {
    normal_ = normal;
}

float3 Receiver::getPosition() const {
    return pos_;
}

void Receiver::setPosition(float3 pos) {
    pos_ = pos;
}

float3 Receiver::getSize() const {
    return size_;
}

void Receiver::setSize(float3 size) {
    size_ = size;
}

int Receiver::getFaceIndex() const {
    return face_num_;
}

void Receiver::setFaceIndex(int face_num) {
    face_num_ = face_num;
}

float Receiver::getPixelLength() const {
    return pixel_length_;
}