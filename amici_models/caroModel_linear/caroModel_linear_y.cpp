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

void y_caroModel_linear(realtype *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    y[0] = x2;
}

} // namespace model_caroModel_linear
} // namespace amici
