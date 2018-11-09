//
// Created by dxt on 18-11-6.
//

#include <stdexcept>

#include "RegularExpressionTree.h"

/**
 * TreeNodes function
 * */
void TreeNode::init_next() {
    for(int i=0;i<NEXT_SIZE;++i) {
        next.push_back(nullptr);
    }
}

TreeNode *TreeNode::getNextNode(char c) {
    int id = c - 'A';
    if(id>=0 && id<NEXT_SIZE) {
        return next[id];
    }
    return nullptr;
}

bool TreeNode::setNextNode(char c, TreeNode *nextNode) {
    int id = c - 'A';
    if(id>=0 && id<NEXT_SIZE) {
        next[id] = nextNode;
        return true;
    }
    return false;
}

bool TreeNode::isTerminated() const {
    return terminated_signal;
}

void TreeNode::setTerminatedSignal(bool terminated_signal) {
    TreeNode::terminated_signal = terminated_signal;
}

/**
 * SceneRegularExpressionTree function
 * */
bool SceneRegularExpressionTree::setUpTree() {
    start_node = new TreeNode(false);
    ground_node = new TreeNode(false);
    receiver_node = new TreeNode(false);
    grid_node = new TreeNode(false);
    helio_node = new TreeNode(true);

    start_node->setNextNode('D', ground_node);
    ground_node->setNextNode('R', receiver_node);
    receiver_node->setNextNode('G', grid_node);
    grid_node->setNextNode('H', helio_node);

    helio_node->setNextNode('R', receiver_node);
    helio_node->setNextNode('G', grid_node);
    helio_node->setNextNode('H', helio_node);

    return true;
}

bool SceneRegularExpressionTree::destroyTree(){
    delete helio_node;
    delete grid_node;
    delete receiver_node;
    delete ground_node;
    delete start_node;

    return true;
}

TreeNode *SceneRegularExpressionTree::step_forward(TreeNode *node, char c) {
    if(!node || !node->getNextNode(c)) {
        throw std::runtime_error("Cannot step forward.");
    }
    return node->getNextNode(c);
}

void SceneRegularExpressionTree::check_terminated(TreeNode *node) {
    if(!node) {
        throw std::runtime_error("Nullptr is not the terminated status of the scene regular expression.");
    }
    if(!node->isTerminated()){
        throw std::runtime_error("Current status is not the terminated status.");
    }
}