//
// Created by dxt on 18-11-20.
//

#ifndef SOLARENERGYRAYTRACING_NEWTYPEHELIOSTATFORTEST_H
#define SOLARENERGYRAYTRACING_NEWTYPEHELIOSTATFORTEST_H

#include "Heliostat.cuh"
#include "global_function.cuh"

class NewTypeHeliostatForTest : public Heliostat {
public:
    virtual int CGetDiscreteMicroHelioOriginsAndNormals(float3 *&d_microhelio_vertexs, float3 *&d_microhelio_normals) {}

    virtual int getSubHelioSize();
    virtual void setSize(float3 size);

    virtual void CGetSubHeliostatVertexes(std::vector<float3> &SubHeliostatVertexes);

    virtual void setSurfaceProperty(const std::vector<float> &surface_property){}
    virtual std::vector<float> getSurfaceProperty();
};


#endif //SOLARENERGYRAYTRACING_NEWTYPEHELIOSTATFORTEST_H
