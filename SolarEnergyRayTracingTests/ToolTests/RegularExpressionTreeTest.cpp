//
// Created by dxt on 18-11-6.
//

#include <iostream>
#include <string>

#include "RegularExpressionTree.h"
#include "gtest/gtest.h"

class RegularExpressionTreeFixture : public ::testing::Test {
public:
    bool check_expression(std::string expression) {
        std::cout<<"The expression is: '"<<expression<<"'"<<std::endl;
        TreeNode *node = sceneTree.getRoot();
        int i = 0;
        try {
            for (; i < expression.size(); ++i) {
                node = sceneTree.step_forward(node, expression[i]);
            }
            sceneTree.check_terminated(node);
            return true;
        } catch (std::runtime_error e) {
            std::cerr << e.what() << " Error occurs at position " << i
                      << " in expresion: " << expression << '.' << std::endl;
            return false;
        }
    }

    SceneRegularExpressionTree sceneTree;
};

TEST_F(RegularExpressionTreeFixture, goodExample) {
    // D(R(GH+)+)+
    std::string goodExample1("DRGH");
    std::string goodExample2("DRGHH");
    std::string goodExample3("DRGHGH");
    std::string goodExample4("DRGHHRGHH");

    EXPECT_TRUE(check_expression(goodExample1));
    EXPECT_TRUE(check_expression(goodExample2));
    EXPECT_TRUE(check_expression(goodExample3));
    EXPECT_TRUE(check_expression(goodExample4));
}

TEST_F(RegularExpressionTreeFixture, badExampleOfIncorrectInput) {
    // D(R(GH+)+)+
    std::string emptyExample("");
    std::string lostExample("DR");
    std::string duplicateExample1("DRRGHGH");
    std::string duplicateExample2("DRGGHHRGHH");

    EXPECT_FALSE(check_expression(emptyExample));
    EXPECT_FALSE(check_expression(lostExample));
    EXPECT_FALSE(check_expression(duplicateExample1));
    EXPECT_FALSE(check_expression(duplicateExample2));
}

TEST_F(RegularExpressionTreeFixture, badExampleOfInvalidInput) {
    // D(R(GH+)+)+
    std::string lostExample("dr");
    std::string duplicateExample1("abc");
    std::string duplicateExample2("***abc");

    EXPECT_FALSE(check_expression(lostExample));
    EXPECT_FALSE(check_expression(duplicateExample1));
    EXPECT_FALSE(check_expression(duplicateExample2));
}