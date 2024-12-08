#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_caroModel_linear {

static constexpr std::array<sunindextype, 3> dwdx_colptrs_caroModel_linear_ = {
    0, 2, 3
};

void dwdx_colptrs_caroModel_linear(SUNMatrixWrapper &dwdx){
    dwdx.set_indexptrs(gsl::make_span(dwdx_colptrs_caroModel_linear_));
}
} // namespace model_caroModel_linear
} // namespace amici
