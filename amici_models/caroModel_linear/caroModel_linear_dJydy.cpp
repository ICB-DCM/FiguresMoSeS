#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include <gsl/gsl-lite.hpp>
#include <algorithm>

#include "caroModel_linear_p.h"
#include "caroModel_linear_k.h"
#include "caroModel_linear_y.h"
#include "caroModel_linear_sigmay.h"
#include "caroModel_linear_my.h"
#include "caroModel_linear_dJydy.h"

namespace amici {
namespace model_caroModel_linear {

void dJydy_caroModel_linear(realtype *dJydy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            dJydy[0] = (-1.0*mobs_x2 + 1.0*obs_x2)/std::pow(sigma_obs_x2, 2);
            break;
    }
}

} // namespace model_caroModel_linear
} // namespace amici
