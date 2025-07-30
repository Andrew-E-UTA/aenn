#pragma once

#include <vector>
#include <cmath>
#include <cinttypes>
#include "tensor.h"

template <typename T>
using func = T(T);

float sigmoidf(float x) {
    return 1 / (1 + expf(-1 * x));
}

float dsigmoidf(float x) {
    return  sigmoidf(x) * (1 - sigmoidf(x));
}

float RelUf(float x) {
    return (x > 0)? x : 0;
}

float dRelUf(float x) {
    return (x > 0)? 1: 0;
}

template <typename T>
std::array<size_t, 2> argMax(const Tensor<T>& arguments) {
    size_t nRows = arguments.dim()[0];
    size_t nCols = arguments.dim()[1];
    T max = arguments.at(0, 0);
    std::array<size_t, 2> maxIndex = {0,0};
    for(size_t r = 0; r < nRows; ++r) {
        for(size_t c = 0; c < nCols; ++c) {
            if (arguments.at(r, c) > max) {
                max = arguments.at(r, c);
                maxIndex = {r, c};
            }
        }
    }    
    return maxIndex;
}

template <typename T>
Tensor<T> softMax(const Tensor<T>& results) {
    size_t nRows = results.dim()[0];
    size_t nCols = results.dim()[1];

    T denom = (T) 0;
    T max = results.at(argMax(results));
    Tensor<T> normalized(results.dim());

    for(size_t r = 0; r < nRows; ++r) {
        for(size_t c = 0; c < nCols; ++c) {
            denom += exp(results.at(r, 0) - max);
        }
    }

    for(size_t r = 0; r < nRows; ++r) {
        for(size_t c = 0; c < nCols; ++c) {
            normalized.at(r, 0) = exp(results.at(r,0) - max) / denom;
        }
    }

    return normalized;
}

template <typename T>
T avg(const Tensor<T>& input) {
    size_t nRows = input.dim()[0];
    size_t nCols = input.dim()[1];
    T sum = 0;
    for(size_t r = 0; r < nRows; ++r) {
        for(size_t c = 0; c < nCols; ++c) {
            sum += input.at(r, c);
        }
    }
    sum /= ((T) nRows * nCols);
    return sum;
}


template <typename T>
struct ActivationFunction {
   func<T>* function;
   func<T>* derivative;
};

template <typename T>
class Model {
public:
    Model(const std::vector<size_t>& desc, const std::vector<ActivationFunction<T>>& layerActivations){
        assert((desc.size() - 1) == layerActivations.size());

        size_t nLayerCount = desc.size();   //Layers + 1 (input)

        acts.reserve(nLayerCount);
        zs.reserve(nLayerCount - 1);

        wts.reserve(nLayerCount - 1);
        wtGrads.reserve(nLayerCount - 1);

        bs.reserve(nLayerCount - 1);
        bGrads.reserve(nLayerCount - 1);

        activations.reserve(nLayerCount - 1);

        nCountPropagated = 0;

        acts.push_back(Tensor<T>(desc.at(0), 1));
        //Iterate through Each Layer (get the nodes in that layer)
        for(size_t nLayer = 1; nLayer < nLayerCount; ++nLayer) {
            acts.push_back(Tensor<T>(desc.at(nLayer), 1));
            bs.push_back(Tensor<T>(desc.at(nLayer), 1).random(-1, 1));
            wts.push_back(Tensor<T>(desc.at(nLayer - 1),desc.at(nLayer)).random(-1, 1));
            activations.push_back(layerActivations.at(nLayer - 1));
        }
    }    

    ~Model() {}

    void print(const char modelName[] = "Model") const {
        size_t nLayerCount = bs.size();
        char name[20];
    
        std::cout << modelName << " = [\n";
        for(size_t nLayer = 0; nLayer < nLayerCount; ++nLayer) {
            snprintf(name, sizeof(name), "Bias%lld", nLayer);
            bs.at(nLayer).print(name);
            
            snprintf(name, sizeof(name), "Weight%lld", nLayer);
            wts.at(nLayer).print(name);
        }
        std::cout << "]" << std::endl;
    }

    /*
        First Word = 0xDEADBEEF //(denotes my nn model savefile)
        Second Word = number of layers (N)
        Rest is the Data:
            First N Tensors are the weights
            Last N  Tensors are the biases
                Each Tensor formatted like:
                    First Word = row
                    Second Word = col
                    Next row*col words (row*col*4 bytes) are the hex representations of floats
    */
    void saveModelParams(const char sFileName[]) const {
        std::ofstream outputFileStream(sFileName);
        assert(outputFileStream != nullptr);

        size_t nLayerCount = wts.size();
     
        for(size_t nLayer = 0; nLayer < nLayerCount; ++nLayer) {
            Tensor<T>& t = wts.at(nLayer);
            size_t nRowCount = t.dim()[0];
            size_t nColCount = t.dim()[1];
            for(size_t r = 0; r < nRowCount; ++r) {
                for(size_t r = 0; r < nRowCount; ++r) {
                    T tVal = t.at(r, c);
                    size_t nByteCount = sizeof(T);
                    for(size_t nByte = 0; nByte < nByteCount; ++nByte) {
                        uint8_t nByteVal = ((tVal && (0xFF << nByteCount*4)) >> (nByteCount * 4));
                        outputFileStream.write(nByteVal);
                    }
                }
            }
        }

        for(size_t nLayer = 0; nLayer < nLayerCount; ++nLayer) {
            
        }
    }

    //Passes inputs through the model (Results are deposited at final activation layer)
    const Tensor<T>& forward(const Tensor<T>& inputs) {
        acts.at(0) = inputs;

        zs.clear();

        //iter through each layer and comput the outputs         
        size_t nLayerCount = acts.size() - 1;
        for(size_t nLayer = 0; nLayer < nLayerCount; ++nLayer) {
            Tensor<T>& x = acts.at(nLayer);
            Tensor<T>& w = wts.at(nLayer);
            Tensor<T>& b = bs.at(nLayer);           
            const Tensor<T>& y = w.transpose() * x + b;
            zs.push_back(y);
            acts.at(nLayer + 1) = activations.at(nLayer).function? y.apply(activations.at(nLayer).function): y;
        }
        return getOutput();
    }

    //Forwards an input through the model and returns the loss Tensor
    Tensor<T> cost(const Tensor<T>& input, const Tensor<T>& expected) {
            Tensor<T> outp = forward(input);    //Outputs Stored at final Activation Layer
            Tensor<T> cost = outp - expected;
            return cost.squared();
    }

    void train(const Tensor<T>& trainingData, const Tensor<T> labels, size_t epochs, float learnRate, bool seeCost = false) {
        size_t nTrainCount = trainingData.dim()[0];
        size_t nOutputs = getOutput().dim()[0];
        Tensor<T> totalLoss(nOutputs, 1);

        for(size_t nIteration = 0; nIteration < epochs; ++nIteration) {
            totalLoss.fill((T)0);
            for(size_t nDataPoint = 0; nDataPoint < nTrainCount; ++nDataPoint) {
                const Tensor<T>& inp = trainingData.row(nDataPoint).transpose();
                const Tensor<T>& exp = labels.row(nDataPoint).transpose();
                //Returns the loss of the function (internally calls forward and deposits predictions at acts.end())

                totalLoss += this->cost(inp, exp);  
                
                //Calculates the gradiants for this iteration and stores internally
                this->backward(exp, learnRate);

            }
            totalLoss = totalLoss / (T)nTrainCount;

            this->apply();

            if(seeCost) {
                T avgCost = totalLoss.average();
                std::cout << "epoch " << nIteration << ": ";
                std::cout << avgCost << '\n';
            }
        }
    }

    void backward(const Tensor<T>& expected, T learnRate) {
        //Number of layers (excluding input)
        size_t nLayerCount = acts.size() - 1;

        std::vector<Tensor<T>> deltas(nLayerCount);
        //calculate Delta for output Layer
        Tensor<T> d = dCost(expected);
        ActivationFunction a = activations.at(nLayerCount - 1);
        Tensor<T> z =  (a.derivative != nullptr) 
                            ? zs.at(nLayerCount - 1).apply(a.derivative) 
                            : zs.at(nLayerCount - 1).fill((T) 1);
        Tensor<T> layerDelta = hadamard(d, z);
        deltas.at(nLayerCount - 1) = layerDelta;

        //Starting from Last Hidden Layer, compute the Deltas for each layer
        for(long long nLayer = nLayerCount - 1 - 1; nLayer >= 0; --nLayer) {
            Tensor<T>& wNext = wts.at(nLayer + 1);
            Tensor<T>& deltaNext = deltas.at(nLayer + 1);

            a = activations.at(nLayer);
            Tensor<T> z = (a.derivative != nullptr) 
                            ? zs.at(nLayer).apply(a.derivative) 
                            : zs.at(nLayer).fill((T) 1);

            layerDelta = hadamard((wNext * deltaNext), z);
            deltas.at(nLayer) = layerDelta;
        }
        //Calculate the Gradiants based on the Deltas for each layer
        for (size_t nLayer = 0; nLayer < nLayerCount; ++nLayer) {
            Tensor<T> wGrad = (acts.at(nLayer) * deltas.at(nLayer).transpose());
            Tensor<T> bGrad = deltas.at(nLayer);

            wGrad *= learnRate;
            bGrad *= learnRate;

            if(nCountPropagated == 0) wtGrads.push_back(wGrad);
            else wtGrads.at(nLayer) += wGrad;

            if(nCountPropagated == 0) bGrads.push_back(bGrad);
            else bGrads.at(nLayer) += bGrad;
        }
        nCountPropagated += 1;
    }

    void apply() {
        size_t nLayerCount = acts.size() - 1;
        for(size_t nLayer = 0; nLayer < nLayerCount; ++nLayer) {
            Tensor<T> bGrad = (bGrads.at(nLayer) / (T) nCountPropagated);
            Tensor<T> wtGrad = (wtGrads.at(nLayer) / (T) nCountPropagated);
            wts.at(nLayer) -= wtGrad;
            bs.at(nLayer) -= bGrad;
        }
        wtGrads.clear();
        bGrads.clear();
        nCountPropagated = 0;
    }

private:
    
    Tensor<T> dCost(const Tensor<T>& expected) {
        return getOutput() - expected;   //Trust? Might not be correct
    }

    //Returns RO reference to final activation layer (stores the result)
    const Tensor<T>& getOutput(void) const {
        return acts.back();
    }

    const Tensor<T>& getInput(void) const {
        return *acts.begin();
    }

    void dumpGrads(const char modelName[] = "Model") {
        size_t nLayerCount = bs.size();
        char name[20];
    
        std::cout << modelName  << " Gradiants" << " = [\n";
        for(size_t nLayer = 0; nLayer < nLayerCount; ++nLayer) {
            snprintf(name, sizeof(name), "Bias%lld", nLayer);
            bGrads.at(nLayer).print(name);
            
            snprintf(name, sizeof(name), "Weight%lld", nLayer);
            wtGrads.at(nLayer).print(name);
        }
        std::cout << "]" << std::endl;
    }
    
    void dumpActivations(const char modelName[] = "Model") {
        size_t nLayerCount = bs.size();
        char name[20];
    
        std::cout << modelName  << " Activations" << " = [\n";
        for(size_t nLayer = 0; nLayer < nLayerCount + 1; ++nLayer) {
            snprintf(name, sizeof(name), "Act%lld", nLayer);
            acts.at(nLayer).print(name);
        }
        std::cout << "]" << std::endl;
    }

    void dumpZs(const char modelName[] = "Model") {
        size_t nLayerCount = bs.size();
        char name[20];
    
        std::cout << modelName  << " Gradiants" << " = [\n";
        for(size_t nLayer = 0; nLayer < nLayerCount; ++nLayer) {
            snprintf(name, sizeof(name), "Z%lld", nLayer);
            zs.at(nLayer).print(name);
        }
        std::cout << "]" << std::endl;
    }

private:
    std::vector<Tensor<T>> acts;        //Output of each Layer (activated)
    std::vector<Tensor<T>> zs;          //Temporary Storage of Pre-activated Outputs
    std::vector<Tensor<T>> bs;          //Biases of each Layer
    std::vector<Tensor<T>> bGrads;      //Temporary Storage of Gradiants for the Biases
    std::vector<Tensor<T>> wts;         //Incoming Weights of eachLayer
    std::vector<Tensor<T>> wtGrads;     //Temporary Storage of Gradiants for the Weights
    std::vector<ActivationFunction<T>>   activations;
    size_t nCountPropagated = 0;
    
};
