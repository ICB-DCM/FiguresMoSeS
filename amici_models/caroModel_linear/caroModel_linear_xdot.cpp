#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "caroModel_linear_x.h"
#include "caroModel_linear_p.h"
#include "caroModel_linear_k.h"
#include "caroModel_linear_w.h"
#include "caroModel_linear_xdot.h"

namespace amici {
namespace model_caroModel_linear {

void xdot_caroModel_linear(realtype *xdot, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    xdot0 = flux_R1 - flux_R2 - flux_R3;  // xdot[0]
    xdot1 = flux_R2 - flux_R3;  // xdot[1]
}

} // namespace model_caroModel_linear
} // namespace amici
