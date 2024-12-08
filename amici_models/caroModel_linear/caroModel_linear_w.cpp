#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "caroModel_linear_x.h"
#include "caroModel_linear_p.h"
#include "caroModel_linear_k.h"
#include "caroModel_linear_w.h"

namespace amici {
namespace model_caroModel_linear {

void w_caroModel_linear(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl){
    flux_R1 = k1;  // w[0]
    flux_R2 = k2*x1;  // w[1]
    flux_R3 = k3*x1*x2;  // w[2]
}

} // namespace model_caroModel_linear
} // namespace amici
