#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "caroModel_linear_x.h"
#include "caroModel_linear_p.h"
#include "caroModel_linear_k.h"

namespace amici {
namespace model_caroModel_linear {

void x_rdata_caroModel_linear(realtype *x_rdata, const realtype *x, const realtype *tcl, const realtype *p, const realtype *k){
    x_rdata[0] = x1;
    x_rdata[1] = x2;
}

} // namespace model_caroModel_linear
} // namespace amici
