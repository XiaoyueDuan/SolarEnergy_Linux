#ifndef SOLARENERGYRAYTRACING_HELIOSTAT_CUH
#define SOLARENERGYRAYTRACING_HELIOSTAT_CUH

#include <cuda_runtime.h>

enum SubCenterType
{
    Square,
    Poisson
};

class Heliostat
{
public:
    Heliostat():subCenterType_(Square){}

    virtual void CRotate(const float3 &focus_center, const float3 &sunray_dir) = 0;

    const float3 &getPosition() const;
    void setPosition(float3 pos_);

    const float3 &getSize() const;
    void setSize(float3 size_);

    const float3 &getNormal() const;
    void setNormal(float3 normal_);

    const int2 &getRowAndColumn() const;
    void setRowAndColumn(int2 row_col_);

    const float2 &getGap() const;
    void setGap(float2 gap_);

    SubCenterType getSubCenterType() const;
    void setSubCenterType(SubCenterType type_);

    float getPixelLength() const;
    void setPixelLength(float pixel_length_);

protected:
    float3 pos_;
    float3 size_;
    float3 normal_;
    int2 row_col_;		// How many mirrors compose a heliostat
    float2 gap_;		// The gap between mirrors
    SubCenterType subCenterType_;
    float pixel_length_;
};

#endif //SOLARENERGYRAYTRACING_HELIOSTAT_CUH