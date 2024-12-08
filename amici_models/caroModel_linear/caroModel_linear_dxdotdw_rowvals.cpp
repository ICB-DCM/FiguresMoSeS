#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_caroModel_linear {

static constexpr std::array<sunindextype, 5> dxdotdw_rowvals_caroModel_linear_ = {
    0, 0, 1, 0, 1
};

void dxdotdw_rowvals_caroModel_linear(SUNMatrixWrapper &dxdotdw){
    dxdotdw.set_indexvals(gsl::make_span(dxdotdw_rowvals_caroModel_linear_));
}
} // namespace model_caroModel_linear
} // namespace amici
