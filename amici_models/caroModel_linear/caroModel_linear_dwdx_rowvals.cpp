#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_caroModel_linear {

static constexpr std::array<sunindextype, 3> dwdx_rowvals_caroModel_linear_ = {
    1, 2, 2
};

void dwdx_rowvals_caroModel_linear(SUNMatrixWrapper &dwdx){
    dwdx.set_indexvals(gsl::make_span(dwdx_rowvals_caroModel_linear_));
}
} // namespace model_caroModel_linear
} // namespace amici
