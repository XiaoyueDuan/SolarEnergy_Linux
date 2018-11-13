//
// Created by dxt on 18-11-2.
//

#ifndef SOLARENERGYRAYTRACING_SCENECONFIGURATION_H
#define SOLARENERGYRAYTRACING_SCENECONFIGURATION_H

#include "cuda_runtime.h"
#include <string>

class SceneConfiguration {
private:
    SceneConfiguration() {
        sun_dir = make_float3(1.0f, 0.0f, 0.0f);
    }
    static SceneConfiguration *sceneConfigurationInstance;

    float3 sun_dir;
    float dni = 1000.0f;
    float csr = 0.1f;
    int num_sunshape_groups = 128;
    int num_sunshape_lights_per_group = 2048;
    int inverse_transform_sampling_groups = 1024;

    float receiver_pixel_length = 0.01f;

    float disturb_std = 0.001f;
    float helio_pixel_length = 0.01f;
    float reflected_rate = 0.88f;

public:
    static SceneConfiguration* getInstance();
    int loadConfiguration(std::string configuration_file_path);

    /**
     *  For tests
     * */
    const float3 &getSun_dir() const;
    float getDni() const;
    float getCsr() const;
    int getNum_sunshape_groups() const;
    int getNum_sunshape_lights_per_group() const;
    int getInverse_transform_sampling_groups() const;

    float getReceiver_pixel_length() const;
    float getDisturb_std() const;
    float getHelio_pixel_length() const;
    float getReflected_rate() const;

    // For tests
    const int num_of_fields = 10;
};


#endif //SOLARENERGYRAYTRACING_SCENECONFIGURATION_H
