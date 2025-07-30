#include <ctime>
#include <iostream>

// #define DEBUG
#include "nn.h"

int main()
{

    srand(101);
    //Forward Declare Activation structs for my type
    ActivationFunction<float> sigf = {sigmoidf, dsigmoidf};
    // ActivationFunction<float> relf = {RelUf, dRelUf};

    //Construct the Model Description
    const std::vector<size_t> layerDesc = {2, 3, 3, 2};
    std::vector<ActivationFunction<float>> layerActivations = {sigf, sigf, sigf};

    //Create Model from the Layer sizes and activations
    Model<float> m(layerDesc, layerActivations);

    //For this specific test, defining training Data
    //Each row is a collection of the inputs/outpus
    //The model will split this based on inparams and outparams
    //Given a Model with X inputs and Y outputs 
    //  First X columns are the input
    //  Last Y columns are the output
    float xor_data[] = {
        0, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 0, 1,
        1, 1, 1, 0
    };
    Tensor<float> data(4, 4, xor_data);

    std::cout << "<<========================>>" << std::endl;
    std::cout << "Training Model..." << std::endl;
    std::cout << "<<========================>>" << std::endl;
    const Tensor<float>& trainData = data.col(0, 2);
    const Tensor<float>& trainLabels = data.col(2, 2);
    m.train(trainData, trainLabels, 100000, 1, false);
    std::cout << "Done!\n" << std::endl;
    
    for(size_t nDataPoint = 0; nDataPoint < 4; ++nDataPoint) {
        const Tensor<float>& inp = data.col(0, 2).row(nDataPoint).transpose();
        const Tensor<float>& exp = data.col(2, 2).row(nDataPoint).transpose();       

        Tensor<float> results = m.forward(inp);
        results = softMax(results);
        std::array<size_t, 2> predictionIndex = argMax(results);

        std::cout << "Input[" << nDataPoint << "] = " << data.col(0, 2).row(nDataPoint);
        std::cout << " Prediction[" << nDataPoint << "] = " << predictionIndex[0] << std::endl;

        // char name[20];
        // snprintf(name, sizeof(name), "Pred[%lld]", nDataPoint);
        // Tensor<float> results = m.forward(inp);
        // results.print(name);

        // results = softMax(results);
        // snprintf(name, sizeof(name), "SoftMax Pred[%lld]", nDataPoint);
        // results.print(name);

        // snprintf(name, sizeof(name), "Avg. Cost[%lld]: ", nDataPoint);
        // Tensor<float> c = m.cost(inp, exp);
        // float avgCost = avg(c);
        // std::cout << name << avgCost << std::endl;
    }

    return 0;
}