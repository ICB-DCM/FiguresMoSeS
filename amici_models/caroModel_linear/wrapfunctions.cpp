#include "amici/model.h"
#include "wrapfunctions.h"
#include "caroModel_linear.h"

namespace amici {
namespace generic_model {

std::unique_ptr<amici::Model> getModel() {
    return std::unique_ptr<amici::Model>(
        new amici::model_caroModel_linear::Model_caroModel_linear());
}


} // namespace generic_model

} // namespace amici
