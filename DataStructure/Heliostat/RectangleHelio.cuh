#ifndef SOLARENERGYRAYTRACING_RECTANGLEHELIO_CUH
#define SOLARENERGYRAYTRACING_RECTANGLEHELIO_CUH

#include "Heliostat.cuh"

class RectangleHelio :public Heliostat
{
public:
    __device__ __host__ RectangleHelio() {}
    virtual void CRotate(const float3 &focus_center, const float3 &sunray_dir);
    virtual void Cget_vertex(float3 &v0, float3 &v1, float3 &v3)
    {
        v0 = vertex_[0];
        v1 = vertex_[1];
        v3 = vertex_[3];
    }

private:
    void Cset_localvertex();
    void Cset_worldvertex();
    void Cset_normal(const float3 &focus_center, const float3 &sunray_dir);

    float3 vertex_[4];
};

#endif //SOLARENERGYRAYTRACING_RECTANGLEHELIO_CUH