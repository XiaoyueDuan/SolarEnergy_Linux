//
// Created by dxt on 18-11-12.
//

#include <iostream>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "global_function.cuh"
#include "RectGrid.cuh"
#include "RectangleHelio.cuh"

vector<int> convert2vector(int *array, int size) {
    vector<int> ans;
    for (int i = 0; i < size; ++i) {
        ans.push_back(array[i]);
    }
    return ans;
}

TEST(CGridHelioMatch, goodExample2) {
    /**
     * The scene is similar in test_file/test_rectgrid.scn
     * */
    RectGrid rectGrid;
    rectGrid.setGridType(0);
    rectGrid.setPosition(make_float3(-15.0f, 0.0f, 20.0f));
    rectGrid.setSize(make_float3(30.0f, 10.0f, 20.0f));
    rectGrid.setInterval(make_float3(10.0f, 10.0f, 10.0f));
    rectGrid.setNumberOfHeliostats(5);
    rectGrid.setHeliostatType(0);
    rectGrid.setStartHeliostatPosition(0);
    rectGrid.Cinit();

    float3 helio_size = make_float3(4.0f, 0.1f, 3.0f);

    Heliostat *h1 = new RectangleHelio();
    h1->setSize(helio_size * 1.5f);
    h1->setPosition(make_float3(-8.0f, 5.0f, 25.0f));

    Heliostat *h2 = new RectangleHelio();
    h2->setSize(helio_size * 0.75f);
    h2->setPosition(make_float3(10.0f, 5.0f, 22.50f));

    Heliostat *h3 = new RectangleHelio();
    h3->setSize(helio_size * 0.75f);
    h3->setPosition(make_float3(10.0f, 5.0f, 27.50f));

    Heliostat *h4 = new RectangleHelio();
    h4->setSize(helio_size);
    h4->setPosition(make_float3(-10.0f, 5.0f, 35.0f));

    Heliostat *h5 = new RectangleHelio();
    h5->setSize(helio_size);
    h5->setPosition(make_float3(0.0f, 5.0f, 35.0f));

    vector<Heliostat *> heliostats;
    heliostats.push_back(h1);
    heliostats.push_back(h2);
    heliostats.push_back(h3);
    heliostats.push_back(h4);
    heliostats.push_back(h5);

    // test function
    rectGrid.CGridHelioMatch(heliostats);

    // check the result as our expect
    int *h_grid_helio_match = nullptr;
    int *h_grid_helio_index = nullptr;

    int *d_grid_helio_match = rectGrid.getDeviceGridHeliostatMatch();
    int *d_grid_helio_index = rectGrid.getDeviceGridHelioIndex();
    int size = rectGrid.getGridNumber().x * rectGrid.getGridNumber().y * rectGrid.getGridNumber().z + 1;

    global_func::gpu2cpu(h_grid_helio_match, d_grid_helio_match, rectGrid.getNumberOfGridHeliostatMatch());
    global_func::gpu2cpu(h_grid_helio_index, d_grid_helio_index, size);

    std::cout << "Matches:" << std::endl;
    EXPECT_THAT(convert2vector(h_grid_helio_index, size), testing::ElementsAreArray({0, 1, 2, 3, 4, 6, 6}));

    std::cout << "Index:" << std::endl;
    EXPECT_THAT(convert2vector(h_grid_helio_match, rectGrid.getNumberOfGridHeliostatMatch()),
                testing::ElementsAreArray({0, 3, 0, 4, 1, 2}));

    // clean
    delete (h1);
    delete (h2);
    delete (h3);
    delete (h4);
    delete (h5);
}
