The project contains the code accompanied by "Estimation of multifactor stochastic volatility jump-diffusion models: A marginalized filter approach", Bégin and Zafari (2025).

It includes the following main scripts for simulation and empirical analysis:

  Simulation:
    - Filtering:
        -- test_sim_filters: simulates paths and converts them to noisy observations which are then used to filter the latent factors. Three methods are used, i.e., KFDNF, KF, and PF.
    - Estimation:
        -- test_sim_estimation_kfdnf: simulates paths and converts them to noisy observations which are then used to estimate the model parameters using KFDNF.
        -- test_sim_estimation_kf: simulates paths and converts them to noisy observations which are then used to estimate the model parameters using KF.

  Empirical:
    - Estimation:
        -- test_emp_estim_kfdnf: uses the moments (obtained from option prices -- retrieved from OptionMetrics -- and converted to risk-neutral moments) to estimate model parameters using KFDNF.
        -- test_emp_estim_kf: uses the moments (obtained from option prices -- retrieved from OptionMetrics -- and converted to risk-neutral moments) to estimate model parameters using KF.

