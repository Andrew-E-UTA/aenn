#include <ctime>
#include <iostream>
#include <fstream>

#include "nn.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

const ActivationFunction<float> sigf = {sigmoidf, dsigmoidf};

#define READ_WORD(num, buf, idx)\
    num &= ~0xFFFFFFFF;\
    num |= (buf[idx++]<<24 & 0xFF000000);\
    num |= (buf[idx++]<<16 & 0x00FF0000);\
    num |= (buf[idx++]<<8  & 0x0000FF00);\
    num |= (buf[idx++]<<0  & 0x000000FF);

//Assumed float type since mnist
void loadIdx3ToVec(std::vector<std::vector<uint8_t>>& vecImages, const char filePath[]) {
    std::string sFileName(filePath);

    //Open fileStream
    std::ifstream ifs (filePath, std::ifstream::binary);
    char *buffer;
    if(!ifs) exit(-1);

    //Read file into char buffer
    ifs.seekg (0, ifs.end); 
    size_t nLength = ifs.tellg(); 
    ifs.seekg (0, ifs.beg); 
    buffer = new char[nLength];
    ifs.read(buffer, nLength);
    if(!ifs)  {
        ifs.close();
        exit(-1);
    }

    //Verify magic number
    uint32_t nMagicNumber = 0;
    size_t nIndex = 0;
    READ_WORD(nMagicNumber, buffer, nIndex);

    //Populate the dimension sizes
    constexpr size_t nNumDims = 3;
    uint32_t nDimSize[nNumDims] = {};
    for(size_t i = 0; i < nNumDims; ++i){
        READ_WORD(nDimSize[i], buffer, nIndex);
    }

    size_t nNumImages = nDimSize[0];
    size_t nNumValsInImage = nDimSize[1] * nDimSize[2];

    vecImages.clear();
    for(size_t image = 0; image < nNumImages; ++image) {
        std::vector<uint8_t> tempVec;
        for(size_t byte = 0; byte < nNumValsInImage; ++byte) {
            tempVec.push_back(buffer[nIndex++]);
        }
        vecImages.push_back(tempVec);
    }    
}

void loadIdx1ToVec(std::vector<uint8_t>& vecLabels, const char filePath[]) {
    std::string sFileName(filePath);

    //Open fileStream
    std::ifstream ifs (filePath, std::ifstream::binary);
    char *buffer;
    if(!ifs) exit(-1);

    //Read file into char buffer
    ifs.seekg (0, ifs.end); 
    size_t nLength = ifs.tellg(); 
    ifs.seekg (0, ifs.beg); 
    buffer = new char[nLength];
    ifs.read(buffer, nLength);
    if(!ifs)  {
        ifs.close();
        exit(-1);
    }

    //Verify magic number
    uint32_t nMagicNumber = 0;
    size_t nIndex = 0;
    READ_WORD(nMagicNumber, buffer, nIndex);

    //Populate the dimension sizes
    uint32_t nNumLabels;
    READ_WORD(nNumLabels, buffer, nIndex);

    vecLabels.clear();
    for(size_t image = 0; image < nNumLabels; ++image) {
        vecLabels.push_back(buffer[nIndex++]);
    }    
}

/*
    Selects nTrainCount random training images and compresses them into a tensor that the NN library expects
        Each respective Row represents a single input image / label
        For this test, destLabels has dim {nTrainCount, 1}
                       destImages has dim {nTrainCount, 784}
*/
void makeTensorSet(size_t nSetCount, std::vector<std::vector<uint8_t>>& srcImages, std::vector<uint8_t>& srcLabels, Tensor<float>& destImages, Tensor<float>& destLabels) {
    constexpr size_t nImageSize = 784;
    constexpr size_t nLabelSize = 10;
    assert(srcImages.size() == srcLabels.size());
    size_t nImageCount = srcLabels.size();

    std::vector<float> vecTrainImagesConcatonated;
    std::vector<float> vecTrainLabelOutputs;

    vecTrainImagesConcatonated.reserve(nImageSize * nSetCount);
    vecTrainLabelOutputs.reserve(nSetCount);

    for(size_t i = 0; i < nSetCount; ++i){
        size_t nTrainImageIndex = rand() % nImageCount;   //select random training image to use in training
        std::vector<uint8_t>& randomImage = srcImages.at(nTrainImageIndex);
        uint8_t label = srcLabels.at(nTrainImageIndex);

        for(size_t j = 0; j < nImageSize; ++j) {
            float value = 1.0f * randomImage.at(j);
            vecTrainImagesConcatonated.push_back(value);
        }

        for(size_t j = 0; j < nLabelSize; ++j) {
            float value = 1.0f * (j == label);
            vecTrainLabelOutputs.push_back(value);
        }
    }
    destImages.resize(nSetCount, nImageSize);
    destImages.copy(vecTrainImagesConcatonated);

    destLabels.resize(nSetCount, nLabelSize);
    destLabels.copy(vecTrainLabelOutputs);
}

int main(void)
{   
    srand(time(0));

    constexpr float fLearnRate = 1e-1;
    constexpr size_t nEpochs = 5000;
    constexpr size_t nTrainCount = 100;
    constexpr char nTestCount = 100;

    //Construct the Model Description
    /*
        MNIST Model:
            784 inputs
            Hidden Layer 1: 16 nodes 
                Activation sigmoid
            Hidden Layer 2: 16 nodes
                Activation sigmoid
            10 outputs
    */
    const std::vector<size_t> layerDesc = {784, 16, 16, 10};
    std::vector<ActivationFunction<float>> layerActivations = {sigf, sigf, sigf};

    //Create Model from the Layer sizes and activations
    Model<float> m(layerDesc, layerActivations);

    //Load Training Images 
    std::vector<std::vector<uint8_t>> vecTrainImages;
    std::vector<uint8_t> vecTrainLabels;
    loadIdx3ToVec(vecTrainImages, "data/train-images.idx3-ubyte");
    loadIdx1ToVec(vecTrainLabels, "data/train-labels.idx1-ubyte");
    
    //Grab Random Subset of training images to train the models with
    Tensor<float> trainingData(nTrainCount, 28*28);
    Tensor<float> trainingOutputs(nTrainCount, 10);
    makeTensorSet(nTrainCount, vecTrainImages, vecTrainLabels, trainingData, trainingOutputs);

    //Load Testing Images 
    std::vector<std::vector<uint8_t>> vecTestImages;
    std::vector<uint8_t> vecTestLabels;
    loadIdx3ToVec(vecTestImages, "data/t10k-images.idx3-ubyte");
    loadIdx1ToVec(vecTestLabels, "data/t10k-labels.idx1-ubyte");

    //Grab Random Subset of test images to verify model against
    Tensor<float> testingData(nTestCount, 28*28);
    Tensor<float> testingOutputs(nTestCount, 10);
    makeTensorSet(nTestCount, vecTestImages, vecTestLabels, testingData, testingOutputs);

    std::cout << "<<========================>>" << std::endl;
    std::cout << "Training Model..." << std::endl;
    std::cout << "<<========================>>" << std::endl;

    m.train(trainingData, trainingOutputs, nEpochs, fLearnRate, true);

    std::cout << "Done!\n" << std::endl;

    stbi_write_png("test.png",28, 28, 1, (vecTestImages.at(nTestCount)).data(), 28);
    std::cout << (int)vecTestLabels.at(nTestCount) << std::endl;

    size_t correct = 0;
    for(size_t nTestImage = 0; nTestImage < nTestCount; ++nTestImage) {
        const Tensor<float>& inp = testingData.row(nTestImage).transpose();
        const Tensor<float>& exp = testingOutputs.row(nTestImage).transpose();

        Tensor<float> actual = m.forward(inp);

        actual = softMax(actual);
        std::array<size_t, 2> expectedIndex = argMax(exp);
        std::array<size_t, 2> predictionIndex = argMax(actual);
        
        std::cout << "Predicted Label[" << nTestImage << "]: " << predictionIndex[0] << '\n';
        std::cout << "Expected Label[" << nTestImage << "]: " << expectedIndex[0] << '\n';
        std::cout << '\n' << std::endl;
        correct += predictionIndex[0] == expectedIndex[0];
    }
    float accuracy = (float)correct / (float)nTestCount;
    std::cout << "Accuracy = " << std::setw(10) << std::fixed << std::setprecision(3) << accuracy << std::endl;

    return 0;
}