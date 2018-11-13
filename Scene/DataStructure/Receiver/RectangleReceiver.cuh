#ifndef SOLARENERGYRAYTRACING_RECTANGLERECEIVER_CUH
#define SOLARENERGYRAYTRACING_RECTANGLERECEIVER_CUH

#include "Receiver.cuh"

class RectangleReceiver :public Receiver
{
public:
    __device__ __host__ RectangleReceiver() {}
    __device__ __host__ RectangleReceiver(const RectangleReceiver &rect_receiver):Receiver(rect_receiver)
    {
        rect_vertex_[0] = rect_receiver.getRectVertex(0);
        rect_vertex_[1] = rect_receiver.getRectVertex(1);
        rect_vertex_[2] = rect_receiver.getRectVertex(2);
        rect_vertex_[3] = rect_receiver.getRectVertex(3);
        localnormal_ = rect_receiver.getLocalNormal();
    }

    __device__ __host__ bool GIntersect(const float3 &orig, const float3 &dir, float &t, float &u, float &v);

    virtual void CInit(int geometry_info);
    const float3 &getRectVertex(int index) const;
    const float3 &getLocalNormal() const;

private:
    void Cinit_vertex();
    void Cset_localnormal();									// set local normal
    void Cset_localvertex();									// set local vertex position
    void Cset_vertex();											// set world vertex
    virtual void Cset_resolution(int geometry_info);
    virtual void Cset_focuscenter();							// call this function after Cset_vertex();

    float3 localnormal_;
    float3 rect_vertex_[4];
};

#endif //SOLARENERGYRAYTRACING_RECTANGLERECEIVER_CUH