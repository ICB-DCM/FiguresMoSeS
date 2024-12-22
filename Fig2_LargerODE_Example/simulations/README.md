# Synthetic model
- Synthetic 4-compartment model from [FAMoS paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007230)
- True model (free parameters: ro_B, ro_C, mu_AB, mu_BC, mu_BD) was used to generate synthetic data with additive normal 
   noise (mean= 0, sd=1) and noise scaling 0.1 .
- model's parameters scales are log10 for the transition rates \ro... 
 and lin for the multiplication rates \mu...
- initial values for \ro... should be sampled in [0.001,5], parameter bounds are [1e-15,5]

parameter interpretation:
- \rho_x : multiplication rate of compartment X
- \mu_{XY} : transition rate from compartment X to Y 

true model: 5 free parameters:  ro_B, ro_C, mu_AB, mu_BC, mu_BD
