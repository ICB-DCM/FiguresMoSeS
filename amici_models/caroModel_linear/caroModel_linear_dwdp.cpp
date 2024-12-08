#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "caroModel_linear_x.h"
#include "caroModel_linear_p.h"
#include "caroModel_linear_k.h"
#include "caroModel_linear_w.h"
#include "caroModel_linear_dwdp.h"

namespace amici {
namespace model_caroModel_linear {

void dwdp_caroModel_linear(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *dtcldp){
    dflux_R1_dk1 = 1;  // dwdp[0]
    dflux_R2_dk2 = x1;  // dwdp[1]
    dflux_R3_dk3 = x1*x2;  // dwdp[2]
}

} // namespace model_caroModel_linear
} // namespace amici
