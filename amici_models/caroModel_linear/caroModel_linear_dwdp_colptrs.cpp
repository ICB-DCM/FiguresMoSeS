#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_caroModel_linear {

static constexpr std::array<sunindextype, 6> dwdp_colptrs_caroModel_linear_ = {
    0, 1, 2, 3, 3, 3
};

void dwdp_colptrs_caroModel_linear(SUNMatrixWrapper &dwdp){
    dwdp.set_indexptrs(gsl::make_span(dwdp_colptrs_caroModel_linear_));
}
} // namespace model_caroModel_linear
} // namespace amici
