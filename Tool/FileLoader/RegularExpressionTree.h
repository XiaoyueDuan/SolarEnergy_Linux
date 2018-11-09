//
// Created by dxt on 18-11-6.
//

#ifndef SOLARENERGYRAYTRACING_REGULAREXPRESSIONTREE_H
#define SOLARENERGYRAYTRACING_REGULAREXPRESSIONTREE_H

#define NEXT_SIZE 26
#include <vector>

class TreeNode {
private:
    bool terminated_signal;
    std::vector<TreeNode *> next;
    void init_next();

public:
    TreeNode():terminated_signal(false) {
        init_next();
    }
    TreeNode(bool terminated):terminated_signal(terminated) {
        init_next();
    }

    TreeNode *getNextNode(char c);
    bool setNextNode(char c, TreeNode *nextNode);

    bool isTerminated() const;
    void setTerminatedSignal(bool terminated_signal);
};

class SceneRegularExpressionTree {
public:
    SceneRegularExpressionTree() {
        setUpTree();
    }

    ~SceneRegularExpressionTree() {
        destroyTree();
    }

    TreeNode *getRoot() {
        return start_node;
    }

    TreeNode *step_forward(TreeNode *node, char c);
    void check_terminated(TreeNode *node);

private:
    bool setUpTree();
    bool destroyTree();

    TreeNode *start_node;
    TreeNode *ground_node;
    TreeNode *receiver_node;
    TreeNode *grid_node;
    TreeNode *helio_node;
};

#endif //SOLARENERGYRAYTRACING_REGULAREXPRESSIONTREE_H
