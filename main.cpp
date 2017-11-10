//
//  main.cpp
//  testtensorcapi
//
//  Created by tkt on 2017/11/05.
//  Copyright © 2017年 toyozaki. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <random>
#include <tensorflow/c/c_api.h>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"

using namespace tensorflow;
using namespace tensorflow::ops;

void create_train_tensor(TTypes<float>::Matrix &x_tensor, TTypes<float>::Matrix &y_tensor);

int main(int argc, const char * argv[]) {
    // insert code here...
    printf("hello from tensorflow c library version %s\n", TF_Version());
    
    printf("start training. \n");
    
    std::string graph_definition = "mlp2.pb";
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    std::vector<Tensor> outputs; // Store outputs
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));
    
    // create a new session
    TF_CHECK_OK(NewSession(opts, &session));
    
    // Load graph into session
    TF_CHECK_OK(session->Create(graph_def));
    
    // Initialize our variables
    TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));
    
    // create train_data.
    int data_num = 100000;
    Tensor x(DT_FLOAT, TensorShape({data_num, 3}));
    Tensor y(DT_FLOAT, TensorShape({data_num, 4}));
    auto _XTensor = x.matrix<float>();
    auto _YTensor = y.matrix<float>();
    create_train_tensor(_XTensor, _YTensor);

    // create test_data
    int test_num = 100;
    Tensor tx(DT_FLOAT, TensorShape({test_num, 3}));
    Tensor ty(DT_FLOAT, TensorShape({test_num, 4}));
    auto _TXT = tx.matrix<float>();
    auto _TYT = ty.matrix<float>();
    create_train_tensor(_TXT, _TYT);
    
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<int> dice(1,100);
    
    std::map<int, std::string> ansmap;
    ansmap[0] = "+";
    ansmap[1] = "-";
    ansmap[2] = "÷";
    ansmap[3] = "×";
    for (int i = 0; i < 2000; ++i) {

        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr));
        
        if (i%100==0) {
            TF_CHECK_OK(session->Run({{"x", tx}, {"y", ty}}, {"loss","y_out","y_argout"}, {}, &outputs));
            float loss = outputs[0].scalar<float>()(0)/test_num;
            std::cout << "epoch : " << i << ",  loss: " << loss << std::endl;
            for (int j=0; j<3; j++) {
                int dc = dice(mt);
                float x0 = tx.matrix<float>()(dc,0);
                float x1 = tx.matrix<float>()(dc,1);
                float x2 = tx.matrix<float>()(dc,2);
                std::cout << " Q." << j << ": " << x0 << " _ " << x1 << " = " << x2 << std::endl;
                std::cout << "   A: " << ansmap[outputs[2].flat<int64>()(dc)] << "  <--  " << std::endl;
            }
            outputs.clear();
            std::cout << std::endl;
        }
    }
    
    session->Close();
    delete session;
    return 0;
}

void create_train_tensor(TTypes<float>::Matrix &x_tensor, TTypes<float>::Matrix &y_tensor){
    int train_num = x_tensor.dimension(0);
    x_tensor.setRandom();
    y_tensor.setZero();
    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<int> dice(1,4);
    for (int i=0; i<train_num; i++) {
        int dc = dice(mt);
        if (dc==0) {
            x_tensor(i,2) = x_tensor(i,0) + x_tensor(i,1);
            y_tensor(i,0) = 1;
        }
        else if (dc==1){
            x_tensor(i,2) = x_tensor(i,0) - x_tensor(i,1);
            y_tensor(i,1) = 1;
        }
        else if (dc==2){
            x_tensor(i,2) = x_tensor(i,0) / x_tensor(i,1);
            y_tensor(i,2) = 1;
        }
        else{
            x_tensor(i,2) = x_tensor(i,0) * x_tensor(i,1);
            y_tensor(i,3) = 1;
        }
    }
}
