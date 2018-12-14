#ifndef SOLARENERGYRAYTRACING_RECEIVER_CUH
#define SOLARENERGYRAYTRACING_RECEIVER_CUH

#include <cuda_runtime.h>

class Receiver {
public:
    /*
     * Initialize the parameters
     */
    virtual void CInit(int geometry_info) = 0;
    virtual void Cset_resolution(int geometry_info) = 0;
    virtual float3 getFocusCenter(const float3 &heliostat_position) = 0;

    /*
     * Allocate the final image matrix
     */
    void Calloc_image();

    /*
     * Clean the final image matrix
     */
    void Cclean_image_content();

    void CClear();

    __device__ __host__ Receiver() : d_image_(nullptr) {}

    __device__ __host__ Receiver(const Receiver &rect) {
        type_ = rect.type_;
        normal_ = rect.normal_;
        pos_ = rect.pos_;
        size_ = rect.size_;
        face_num_ = rect.face_num_;
        pixel_length_ = rect.pixel_length_;
        d_image_ = rect.d_image_;
        resolution_ = rect.resolution_;
    }

    __device__ __host__ ~Receiver();

    int getType() const;
    void setType(int type_);

    float3 getNormal() const;
    void setNormal(float3 normal);

    float3 getPosition() const;
    void setPosition(float3 pos);

    float3 getSize() const;
    void setSize(float3 size_);

    int getFaceIndex() const;
    void setFaceIndex(int face_num);

    float getPixelLength() const;

    __host__ __device__ float *getDeviceImage() const {
        return d_image_;
    }

    __host__ __device__ int2 getResolution() const {
        return resolution_;
    }

protected:
    int type_;
    float3 normal_;
    float3 pos_;
    float3 size_;
    int face_num_;                        // the number of receiving face
    float pixel_length_;
    float *d_image_;                    // on GPU, size = resolution_.x * resolution_.y
    int2 resolution_;                    // resolution.x is columns, resolution.y is rows
};

#endif //SOLARENERGYRAYTRACING_RECEIVER_CUH