#include <array>
#include <amici/defines.h>

namespace amici {

namespace model_caroModel_linear {

std::array<const char*, 5> parameterNames = {
    "k1", // p[0]
"k2", // p[1]
"k3", // p[2]
"sigma_x2", // p[3]
"noiseParameter1_obs_x2", // p[4]
};

std::array<const char*, 1> fixedParameterNames = {
    "observable_x2", // k[0]
};

std::array<const char*, 2> stateNames = {
    "x1", // x_rdata[0]
"x2", // x_rdata[1]
};

std::array<const char*, 1> observableNames = {
    "", // y[0]
};

std::array<const ObservableScaling, 1> observableScalings = {
    ObservableScaling::lin, // y[0]
};

std::array<const char*, 3> expressionNames = {
    "flux_R1", // w[0]
"flux_R2", // w[1]
"flux_R3", // w[2]
};

std::array<const char*, 5> parameterIds = {
    "k1", // p[0]
"k2", // p[1]
"k3", // p[2]
"sigma_x2", // p[3]
"noiseParameter1_obs_x2", // p[4]
};

std::array<const char*, 1> fixedParameterIds = {
    "observable_x2", // k[0]
};

std::array<const char*, 2> stateIds = {
    "x1", // x_rdata[0]
"x2", // x_rdata[1]
};

std::array<const char*, 1> observableIds = {
    "obs_x2", // y[0]
};

std::array<const char*, 3> expressionIds = {
    "flux_R1", // w[0]
"flux_R2", // w[1]
"flux_R3", // w[2]
};

std::array<int, 2> stateIdxsSolver = {
    0, 1
};

std::array<bool, 0> rootInitialValues = {
    
};

} // namespace model_caroModel_linear

} // namespace amici
