% This script generates paths of noisy observations and then use them to
% filter the latent factors with different methods (KFDNF proposed by Bégin and Zafari (2025), 
% KF used by Feunou and Okou (2016), and particle filter)

clc;clear all
for id=1:1%500
rng(id);

%% Variables
T                       =  252*10;
S0                      =  1000;
MATURITIES              =  [21,63,126,252]/252;
nbParticles             =  100000;
nbK                     =  5;

%% Model creation
model                   = AFT('scale_pvalues',true);
      
%% Initial parameters
initial = {'eta',  0, ...
     'kappa1',  0.5, ...
        'nu1',  0.02, ...
     'sigma1',  0.10, ...
        'mu1',  0.20, ...
       'rho1',  -0.98, ...
        'v10',  1.00e-04, ...
     'kappa2',  0.3, ...
        'nu2',  0.04, ...
     'sigma2',  0.15, ...
       'rho2',  -0.9897, ...
        'v20',  3.000e-04, ...
     'kappa3',  50, ...
        'mu3',  0.6, ...
       'rho3',  0.06, ...
        'v30',  0.005, ...
    'lambdap',  240, ...
    'lambdan',  7, ...
        'cp0',  5, ...
        'cp1',  245, ...
        'cp2',  100, ...
        'cp3',  0, ...
        'cn0',  0, ...
        'cn1',  20, ...
        'cn2',  0.1, ...
        'cn3',  160, ...
       'del2',  0.3, ...
       'del3',  0.3, ...
       'del4',  0.3};
model.setParameterValues(initial{:})
model.params.eta.fixed = 1;
model.params.cn0.fixed = 1;
model.params.cp3.fixed = 1;

pv                                 = model.getPV;

[S,R,h1,h2,h3,xp,xn,y]             = model.simulatePaths(1,T,S0,pv.v10,pv.v20,pv.v30);
model.moments.Maturity             = MATURITIES;
[Volatility,Skewness,Kurtosis]     = model.getMoments(h1(2:end),h2(2:end),h3(2:end),true);
Variance = Volatility.^2;

% calculate the noisy cumulants
cum2                    = Variance + normrnd(zeros(size(Variance)),repmat(pv.del2.*std(Variance),T,1));
cum3                    = Skewness.*Variance.^1.5 + normrnd(zeros(size(Skewness)),repmat(pv.del3.*std(Skewness.*Variance.^1.5),T,1));
cum4                    = (Kurtosis-3).*Variance.^2   + normrnd(zeros(size(Kurtosis)),repmat(pv.del4.*std((Kurtosis-3).*Variance.^2),  T,1));

% calculate the noisy moments from the noisy cumulants
moments.Variance        = cum2;
moments.Skewness        = cum3./cum2.^1.5;
moments.Kurtosis        = 3 + cum4./cum2.^2;
moments.Maturity        = MATURITIES;

% copy series to model object
model.calendar          = (1:T)';
model.series            = R(2:end)';
model.moments.Maturity  = MATURITIES;
model.moments.Variance  = moments.Variance(:,1:4);
model.moments.Skewness  = moments.Skewness(:,1:4);
model.moments.Kurtosis  = moments.Kurtosis(:,1:4);

%% filter with KFDNF method as proposed by Bégin and Zafari (2025)
for dn = 1:10
  dn
  kfdnf                      = KFDNF(model, dn);
  
  tic();
  [LLk{dn},filtk{dn}]        = kfdnf.logLikelihood();
  sum(LLk{dn})
  timek(dn)                  = toc();
end

%% filter with KF method as used by Feunou and Okou (2016)
kf                          = KF(model);
tic(); 
[LLf,filtf]                 = kf.logLikelihood();
sum(LLf)
timef                       = toc();

%% filter with particle filter 
pf                      = ParticleFilterSIR(model,nbParticles,'Stratified');
tic();
[LLp,filtp]             = pf.logLikelihood();
sum(LLp)
timep                   = toc();

series.R                = R;
series.h1               = h1;
series.h2               = h2;
series.h3               = h3;
series.xp               = xp;
series.xn               = xn;
series.y                = y;


save(['results/Filtering/Simulation/filter_simulated_path_',int2str(id)],'series','model','LLk','filtk','timek','LLf','filtf','timef','LLp','filtp','timep')

end


