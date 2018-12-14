//
// Created by dxt on 18-12-13.
//

#include <iostream>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "CylinderReceiver.cuh"
#include "DataStructureUtil.h"

class CylinderReceiverFixture : public ::testing::Test {
protected:
    void SetUp() {
        cylinderReceiver.setType(1);
        cylinderReceiver.setSize(make_float3(3.0f, 4.0f, 0.0f));
        cylinderReceiver.setNormal(make_float3(0.0f, 1.0f, 0.0f));
        cylinderReceiver.setPosition(make_float3(0.0f, 0.0f, 0.0f));
        cylinderReceiver.setFaceIndex(0);
    }

public:
    CylinderReceiver cylinderReceiver;
};

TEST_F(CylinderReceiverFixture, hasIntersection) {
    float3 ans; // t, u, v

    // 1. Origin 1
    float3 origin1 = make_float3(1.5f, 0.0f, -6.0f);
    float3 dirZPositive = make_float3(0.0f, 0.0f, 1.0f);
    EXPECT_TRUE(cylinderReceiver.GIntersect(origin1, dirZPositive, ans.x, ans.y, ans.z));
    EXPECT_TRUE(Float3Eq(make_float3(6 - 1.5 * sqrtf(3), 0.5f, 5.0f / 6.0f), ans, 1e-4));     // expect u: 0.5, v: 5/6

    EXPECT_TRUE(cylinderReceiver.GIntersect(origin1, normalize(make_float3(-1.0f, 0.0f, 2.0f)), ans.x, ans.y, ans.z));
    EXPECT_TRUE(Float3Eq(make_float3(1.5 * sqrtf(5), 0.5f, 0.75f), ans, 1e-4));           // expect u: 0.5, v: 0.75

    // 2. Origin 2
    float3 origin2 = make_float3(-6.0f, 0.0f, -1.5f);
    float3 dirXPositive = make_float3(1.0f, 0.0f, 0.0f);
    EXPECT_TRUE(cylinderReceiver.GIntersect(origin2, dirXPositive, ans.x, ans.y, ans.z));
    EXPECT_TRUE(Float3Eq(make_float3(6 - 1.5 * sqrtf(3), 0.5f, 7.0f / 12.0f), ans, 1e-4));    // expect u: 0.5, v: 7/12

    // 3. Origin 3
    float3 origin3 = make_float3(3.0f, 0.0f, 3.0f);
    float3 dirXNegetive = make_float3(-1.0f, 0.0f, 0.0f);
    EXPECT_TRUE(cylinderReceiver.GIntersect(origin3, dirXNegetive, ans.x, ans.y, ans.z));
    EXPECT_TRUE(Float3Eq(make_float3(3.0f, 0.5f, 0.25f), ans, 1e-4));
    std::cout << ans.x << ' ' << ans.y << ' ' << ans.z << std::endl;                    // expect u: 0.5, v: 0.25

    float3 intersection_point = make_float3(2.1213f, 1.0f, 2.1213f);
    float3 dir2Point = normalize(intersection_point - origin3);
    EXPECT_TRUE(cylinderReceiver.GIntersect(origin3, dir2Point, ans.x, ans.y, ans.z));
    float expect_distance = (3 * sqrtf(2) - 3) / length(make_float2(dir2Point.x, dir2Point.z));
    EXPECT_TRUE(Float3Eq(make_float3(expect_distance, 0.75f, 0.125f), ans, 1e-4));      // expect u: 0.75, v: 0.125
}

TEST_F(CylinderReceiverFixture, noIntersection) {
    cylinderReceiver.setPosition(make_float3(2.0f, 2.0f, 0.0f));

    float3 ans;
    // 1. Inner to the cylinder
    float3 inner_origin1 = make_float3(1.0f);
    float3 dirXPositive = make_float3(1.0f, 0.0f, 0.0f);
    EXPECT_FALSE(cylinderReceiver.GIntersect(inner_origin1, dirXPositive, ans.x, ans.y, ans.z));

    // 2. The intersection location of infinite-cylinder is lower than the lower bound of cylinder
    float3 outter_origin2 = make_float3(-1.0f, -2.0f, -3.0f);
    float3 dirPositive = normalize(make_float3(1.0f, 1.0f, 1.0f));
    EXPECT_FALSE(cylinderReceiver.GIntersect(outter_origin2, dirXPositive, ans.x, ans.y, ans.z));
    EXPECT_FALSE(cylinderReceiver.GIntersect(outter_origin2, dirPositive, ans.x, ans.y, ans.z));

    // 3. The ray direction is far away from the Ro(cosine<0)
    float3 outter_origin3 = make_float3(-1.0f, 0.5f, -3.0f);
    float3 dir = normalize(make_float3(-0.1f, 0.0f, 1.0f));
    EXPECT_FALSE(cylinderReceiver.GIntersect(outter_origin3, dir, ans.x, ans.y, ans.z));
}

TEST_F(CylinderReceiverFixture, getOutterFocusCenter) {
    cylinderReceiver.setPosition(make_float3(2.0f, 3.0f, 1.0f));
    float y = 3.0f;
    float radius = cylinderReceiver.getSize().x;

    float3 heliostat_pos = make_float3(5.0f, 0.0f, -2.0f);
    float3 expect_ans = make_float3(radius * sqrtf(2) / 2 + 2, y, 1 - radius * sqrtf(2) / 2);
    EXPECT_TRUE(Float3Eq(cylinderReceiver.getFocusCenter(heliostat_pos), expect_ans, 1e-4));

    heliostat_pos = make_float3(8.0f, 4.0f, 1.0f);
    expect_ans = make_float3(5.0f, y, 1.0f);
    EXPECT_TRUE(Float3Eq(cylinderReceiver.getFocusCenter(heliostat_pos), expect_ans, 1e-4));

    heliostat_pos = make_float3(-4.0f, 6.0f, 2.0f);
    float2 dir = normalize(make_float2(6.0f, -1.0f));
    float distance =
            length(make_float2(heliostat_pos.x - cylinderReceiver.getPosition().x,
                               heliostat_pos.z - cylinderReceiver.getPosition().z)) - radius;
    expect_ans = make_float3(heliostat_pos.x + distance * dir.x, y, heliostat_pos.z + distance * dir.y);
    EXPECT_TRUE(Float3Eq(cylinderReceiver.getFocusCenter(heliostat_pos), expect_ans, 1e-4));

    heliostat_pos = make_float3(-4.0f, 1.0f, 0.0f);
    expect_ans.z = 2 - expect_ans.z; // symmetry with z=1
    EXPECT_TRUE(Float3Eq(cylinderReceiver.getFocusCenter(heliostat_pos), expect_ans, 1e-4));
}

TEST_F(CylinderReceiverFixture, getInnerFocusCenter) {
    float3 heliostat_pos = make_float3(1.0f);
    EXPECT_TRUE(Float3Eq(cylinderReceiver.getFocusCenter(heliostat_pos), cylinderReceiver.getPosition(), 1e-4));
}

TEST_F(CylinderReceiverFixture, Cset_resolution) {
    cylinderReceiver.Cset_resolution(2);
    EXPECT_EQ(cylinderReceiver.getResolution().x, 38);
    EXPECT_EQ(cylinderReceiver.getResolution().y, 8);
}



