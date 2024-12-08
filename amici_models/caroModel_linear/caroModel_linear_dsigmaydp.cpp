#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "caroModel_linear_p.h"
#include "caroModel_linear_k.h"
#include "caroModel_linear_y.h"

namespace amici {
namespace model_caroModel_linear {

void dsigmaydp_caroModel_linear(realtype *dsigmaydp, const realtype t, const realtype *p, const realtype *k, const realtype *y, const int ip){
    switch(ip) {
        case 4:
            dsigmaydp[0] = 1;
            break;
    }
}

} // namespace model_caroModel_linear
} // namespace amici
