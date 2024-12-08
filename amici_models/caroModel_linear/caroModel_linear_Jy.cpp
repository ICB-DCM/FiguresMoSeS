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

namespace amici {
namespace model_caroModel_linear {

void Jy_caroModel_linear(realtype *Jy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            Jy[0] = 0.5*std::log(2*amici::pi*std::pow(sigma_obs_x2, 2)) + 0.5*std::pow(-mobs_x2 + obs_x2, 2)/std::pow(sigma_obs_x2, 2);
            break;
    }
}

} // namespace model_caroModel_linear
} // namespace amici
