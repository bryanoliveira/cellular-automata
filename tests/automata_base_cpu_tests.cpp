#include "gtest/gtest.h"

insert_glider(int row, int col);

class AutomataBaseCPUTest : public ::testing::Test {
  protected:
    void SetUp() override {
        q1_.Enqueue(1);
        q2_.Enqueue(2);
        q2_.Enqueue(3);
    }

    // void TearDown() override {}

    Queue<int> q0_;
    Queue<int> q1_;
    Queue<int> q2_;
};

TEST_F(AutomataBaseCPUTest, IsEmptyInitially) { EXPECT_EQ(q0_.size(), 0); }