#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_caroModel_linear {

static constexpr std::array<std::array<sunindextype, 1>, 1> dJydy_rowvals_caroModel_linear_ = {{
    {0}, 
}};

void dJydy_rowvals_caroModel_linear(SUNMatrixWrapper &dJydy, int index){
    dJydy.set_indexvals(gsl::make_span(dJydy_rowvals_caroModel_linear_[index]));
}
} // namespace model_caroModel_linear
} // namespace amici
