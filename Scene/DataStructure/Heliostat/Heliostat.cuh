#ifndef SOLARENERGYRAYTRACING_HELIOSTAT_CUH
#define SOLARENERGYRAYTRACING_HELIOSTAT_CUH

#include <cuda_runtime.h>

enum SubCenterType {
    Square,
    Poisson
};

class Heliostat {
public:
    Heliostat() : subCenterType_(Square), sub_helio_size(1) {}

    virtual void CSetNormalAndRotate(const float3 &focus_center, const float3 &sunray_dir);
    virtual int
    CGetDiscreteMicroHelioOriginsAndNormals(float3 *&d_microhelio_vertexs, float3 *&d_microhelio_normals) = 0;
    virtual int getSubHelioSize() = 0;
    /**
     * Only for test. Never use 'setSubHelioSize' standalone
     * */
    virtual void setNumberOfSubHelio(int n) {
        sub_helio_size = n;
    }

    int* generateDeviceMicrohelioGroup(int num_group, int size);

    float3 getPosition() const;
    void setPosition(float3 pos_);

    float3 getSize() const;
    virtual void setSize(float3 size_) = 0;

    float3 getNormal() const;
    void setNormal(float3 normal_);

    int2 getRowAndColumn() const;
    void setRowAndColumn(int2 row_col_);

    float2 getGap() const;
    void setGap(float2 gap_);

    SubCenterType getSubCenterType() const;
    void setSubCenterType(SubCenterType type_);

    float getPixelLength() const;
    void setPixelLength(float pixel_length_);

    int getBelongingGridId() const;
    void setBelongingGridId(int belonging_grid_id_);

    void Cget_vertex(float3 &v0, float3 &v1, float3 &v3);

protected:
    float3 pos_;
    float3 size_;
    float3 vertex_[4];
    float3 normal_;
    int2 row_col_;          // How many mirrors compose a heliostat
    float2 gap_;            // The gap between mirrors
    SubCenterType subCenterType_;
    float pixel_length_;
    int sub_helio_size;
    int belonging_grid_id_;

private:
    void CSetWorldVertex();
    void CSetNormal(const float3 &focus_center, const float3 &sunray_dir);
};

#endif //SOLARENERGYRAYTRACING_HELIOSTAT_CUH