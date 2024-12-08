#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_caroModel_linear {

static constexpr std::array<std::array<sunindextype, 2>, 1> dJydy_colptrs_caroModel_linear_ = {{
    {0, 1}, 
}};

void dJydy_colptrs_caroModel_linear(SUNMatrixWrapper &dJydy, int index){
    dJydy.set_indexptrs(gsl::make_span(dJydy_colptrs_caroModel_linear_[index]));
}
} // namespace model_caroModel_linear
} // namespace amici
