#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_caroModel_linear {

static constexpr std::array<sunindextype, 3> dwdp_rowvals_caroModel_linear_ = {
    0, 1, 2
};

void dwdp_rowvals_caroModel_linear(SUNMatrixWrapper &dwdp){
    dwdp.set_indexvals(gsl::make_span(dwdp_rowvals_caroModel_linear_));
}
} // namespace model_caroModel_linear
} // namespace amici
