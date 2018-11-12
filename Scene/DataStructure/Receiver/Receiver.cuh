#ifndef SOLARENERGYRAYTRACING_RECEIVER_CUH
#define SOLARENERGYRAYTRACING_RECEIVER_CUH

#include <cuda_runtime.h>

class Receiver {
public:
    /*
     * Whether the light intersect with receiver
     * Note: sub-class needs to redefine it
     */
    __device__ __host__ bool GIntersect(const float3 &orig, const float3 &dir,
                                        float &t, float &u, float &v) {
        return true;
    }

    /*
     * Initialize the parameters
     */
    virtual void CInit(int geometry_info) {}
    virtual void Cset_resolution(int geometry_info) {}
    virtual void Cset_focuscenter() {}

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
        focus_center_ = rect.focus_center_;
        face_num_ = rect.face_num_;
        pixel_length_ = rect.pixel_length_;
        d_image_ = rect.d_image_;
        resolution_ = rect.resolution_;
    }

    __device__ __host__ ~Receiver();

    int getType() const;
    void setType(int type_);

    const float3 &getNormal() const;
    void setNormal(float3 normal_);

    const float3 &getPosition() const;
    void setPosition(float3 pos_);

    const float3 &getSize() const;
    void setSize(float3 size_);

    const float3 &getFocusCenter() const;
    void setFocusCenter(float3 focus_center_);

    int getFaceIndex() const;
    void setFaceIndex(int face_num_);

    float getPixelLength() const;
    void setPixelLength(float pixel_length_);

    float *getDeviceImage() const;
    void setDeviceImage(float *d_image_);

    const int2 &getResolution() const;
    void setResolution(int2 resolution_);

protected:
    int type_;
    float3 normal_;
    float3 pos_;
    float3 size_;
    float3 focus_center_;                // fixed for a scene
    int face_num_;                        // the number of receiving face
    float pixel_length_;
    float *d_image_;                    // on GPU, size = resolution_.x * resolution_.y
    int2 resolution_;                    // resolution.x is columns, resolution.y is rows
};

#endif //SOLARENERGYRAYTRACING_RECEIVER_CUH