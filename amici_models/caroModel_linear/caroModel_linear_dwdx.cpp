#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "caroModel_linear_x.h"
#include "caroModel_linear_p.h"
#include "caroModel_linear_k.h"
#include "caroModel_linear_w.h"
#include "caroModel_linear_dwdx.h"

namespace amici {
namespace model_caroModel_linear {

void dwdx_caroModel_linear(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl){
    dflux_R2_dx1 = k2;  // dwdx[0]
    dflux_R3_dx1 = k3*x2;  // dwdx[1]
    dflux_R3_dx2 = k3*x1;  // dwdx[2]
}

} // namespace model_caroModel_linear
} // namespace amici
