#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "caroModel_linear_x.h"
#include "caroModel_linear_p.h"
#include "caroModel_linear_k.h"
#include "caroModel_linear_w.h"
#include "caroModel_linear_dxdotdw.h"

namespace amici {
namespace model_caroModel_linear {

void dxdotdw_caroModel_linear(realtype *dxdotdw, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    dxdot0_dflux_R1 = 1;  // dxdotdw[0]
    dxdot0_dflux_R2 = -1;  // dxdotdw[1]
    dxdot1_dflux_R2 = 1;  // dxdotdw[2]
    dxdot0_dflux_R3 = -1;  // dxdotdw[3]
    dxdot1_dflux_R3 = -1;  // dxdotdw[4]
}

} // namespace model_caroModel_linear
} // namespace amici
