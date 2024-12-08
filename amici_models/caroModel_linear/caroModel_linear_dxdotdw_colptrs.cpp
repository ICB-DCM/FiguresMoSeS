#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_caroModel_linear {

static constexpr std::array<sunindextype, 4> dxdotdw_colptrs_caroModel_linear_ = {
    0, 1, 3, 5
};

void dxdotdw_colptrs_caroModel_linear(SUNMatrixWrapper &dxdotdw){
    dxdotdw.set_indexptrs(gsl::make_span(dxdotdw_colptrs_caroModel_linear_));
}
} // namespace model_caroModel_linear
} // namespace amici
