#ifndef SOLARENERGYRAYTRACING_RECTANGLEHELIO_CUH
#define SOLARENERGYRAYTRACING_RECTANGLEHELIO_CUH

#include "Heliostat.cuh"

class RectangleHelio : public Heliostat {
public:
    RectangleHelio() {}

    virtual int getSubHelioSize() {
        return sub_helio_size;
    }

    virtual int CGetDiscreteMicroHelioOriginsAndNormals(float3 *&d_microhelio_vertexs, float3 *&d_microhelio_normals);
};

#endif //SOLARENERGYRAYTRACING_RECTANGLEHELIO_CUH