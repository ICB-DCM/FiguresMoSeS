## the model selection in general
true model: 01000100001000010010000000010001
     {<Criterion.NLLH: 'NLLH'>: 16.1570009597022,
    <Criterion.BICC: 'BIC'>: 76.5494346194957,
    <Criterion.AIC: 'AIC'>: 48.3140019194043,
    <Criterion.AICC: 'AICc'>: 48.9065945119969}
			

best model found with 1. backward search: M_00010100100000000000001100001000
    {<Criterion.NLLH: 'NLLH'>: 18.364642470051745,
    <Criterion.BIC: 'BIC'>: 75.43528855268346,
    <Criterion.AIC: 'AIC'>: 50.72928494010349,
    <Criterion.AICC: 'AICc'>: 51.188301333546114}
best model found with 1. forward search : M_01000100001000100010000011000100
    {<Criterion.NLLH: 'NLLH'>: 17.41738610539724,
     <Criterion.BIC: 'BIC'>: 84.5996339983973,
     <Criterion.AIC: 'AIC'>: 52.83477221079448,
     <Criterion.AICC: 'AICc'>: 53.57857386368705}


initial local search (report file): custom FAMoS routine and AICc as the criterion


## remaining global search
- exclusion results plausible: for model_sizes m \in {0,1,...7}: #remaining models + # excluded models = math.comb(32,m) 
- 19.27% of model space remain to be checked after initial famos search
  - remaining free models 870451
  - \# models <= 7 parameters = 4514873
- from calibrating the remaining models of size 6 we can exclude all remaining models of size 5 and smaller


## results figures
- model count 
  - total in model space: 4294967296
  - calibrated during our searches: 
    - initial search: 1077
    - exhaustive step: 709577
    - total: 710654
  -> Our method calibrates 99.98 % fewer models.
 
- computation time 
  - during our searches
    - initial search: 13559.019634008408
    - exhaustive step: 2313138.7394206524
    - total: 2326697.759054661
  - extrapolate over entire model space: 
    - average calibration time per model: 3.274023306777505
    - computation time "classic bruteforce": 14061823028.951159
  -> Our method is 99.98% faster.