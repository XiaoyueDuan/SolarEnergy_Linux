#include "Receiver.cuh"
#include "check_cuda.h"
#include "global_function.cuh"

Receiver::~Receiver()
{
    if (d_image_)
        d_image_ = nullptr;
}

void Receiver::CClear()
{
    if (d_image_)
    {
        cudaFree(d_image_);
        d_image_ = nullptr;
    }
}

void Receiver::Calloc_image()
{
    checkCudaErrors(cudaMalloc((void **)&d_image_, sizeof(float)*resolution_.x*resolution_.y));
}

void Receiver::Cclean_image_content()
{
    int n_resolution = resolution_.x*resolution_.y;
    float *h_clean_receiver = new float[n_resolution];
    for (int i = 0; i < n_resolution; ++i)
        h_clean_receiver[i] = 0.0f;

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

void Receiver::setType(int type_) {
    Receiver::type_ = type_;
}

const float3 &Receiver::getNormal() const {
    return normal_;
}

void Receiver::setNormal(float3 normal_) {
    Receiver::normal_ = normal_;
}

const float3 &Receiver::getPosition() const {
    return pos_;
}

void Receiver::setPosition(float3 pos_) {
    Receiver::pos_ = pos_;
}

const float3 &Receiver::getSize() const {
    return size_;
}

void Receiver::setSize(float3 size_) {
    Receiver::size_ = size_;
}

const float3 &Receiver::getFocusCenter() const {
    return focus_center_;
}

void Receiver::setFocusCenter(float3 focus_center_) {
    Receiver::focus_center_ = focus_center_;
}

int Receiver::getFaceIndex() const {
    return face_num_;
}

void Receiver::setFaceIndex(int face_num_) {
    Receiver::face_num_ = face_num_;
}

float Receiver::getPixelLength() const {
    return pixel_length_;
}

void Receiver::setPixelLength(float pixel_length_) {
    Receiver::pixel_length_ = pixel_length_;
}

float *Receiver::getDeviceImage() const {
    return d_image_;
}

void Receiver::setDeviceImage(float *d_image_) {
    Receiver::d_image_ = d_image_;
}

const int2 &Receiver::getResolution() const {
    return resolution_;
}

void Receiver::setResolution(int2 resolution_) {
    Receiver::resolution_ = resolution_;
}