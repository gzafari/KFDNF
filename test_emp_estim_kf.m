clear all; clc; startup;
id = 12;
rng(id)
startup;

%% Variables
MATURITIES              =  [21,63,126,252]/252;
%NBPARTICLES             =  25000;
LASTDAY                 =  6036;
STARTDAY                =  2;


%% Model creation
model = AFT('scale_pvalues',true);


%% Initial parameters
initial = {'eta', 0,...
    'kappa1', 0.4081,...
    'nu1', 0.0125,...
    'sigma1', 0.1004,...
    'mu1', 0.0135,...
    'rho1', -0.9798,...
    'v10', 7.3174e-05,...
    'kappa2', 0.1992,...
    'nu2', 0.0868,...
    'sigma2', 0.1169,...
    'rho2', -0.9896,...
    'v20', 6.6331e-04,...
    'kappa3', 46.8533,...
    'mu3', 0.5630,...
    'rho3', 0.1068,...
    'v30', 1.1257e-04,...
    'lambdap', 231.6707,...
    'lambdan', 5.8350,...
    'cp0', 4.8429,...
    'cp1', 248.0810,...
    'cp2', 116.0724,...
    'cp3', 0,...
    'cn0', 0,...
    'cn1', 13.6764,...
    'cn2', 0.0353,...
    'cn3', 247.4728,...
    'del2', 0.1646,...
    'del3', 0.2277,...
    'del4', 0.2693};

model.params.eta.fixed = 1;
model.params.cn0.fixed = 1;
model.params.cp3.fixed = 1;
model.setParameterValues(initial{:})


model.params.v10.fixed  = 1;
model.params.v20.fixed  = 1;
model.params.v30.fixed  = 1;
model.params.del2.fixed = 1;
model.params.del3.fixed = 1;
model.params.del4.fixed = 1;



% read data
data                        = load('data/ObsMom.mat');
moments                     = data.moments;
model.calendar              = data.dates(STARTDAY:LASTDAY);
model.series                = data.R(STARTDAY:LASTDAY) - data.r(STARTDAY:LASTDAY);
model.moments.Maturity      = MATURITIES;
model.moments.Variance      = (moments.Vol(STARTDAY:LASTDAY,1:4));%./MATURITIES;
model.moments.Skewness      = moments.Skew(STARTDAY:LASTDAY,1:4);
model.moments.Kurtosis      = moments.Kurt(STARTDAY:LASTDAY,1:4);


% consider noisy parameters as an initial parameters for KFDNF estimation 
model.setPValues(model.getPValues + 0.05.*normrnd(0,1,size(model.getPValues)));


% find the unconditional states and use these as initial values
pv                          = model.getPV
[uncMean, uncVar]           = model.unconditionalMean(pv);

% create a model object
kf                          = KF(model);


% model estimation
for di = 1:30
    di
    model.optimopt.MaxIter    = 500;
    results                   = model.maximumLikelihood('fminsearch',kf);
    results                   = model.maximumLikelihood('fmincon',kf);
    
    save(['results/estimation/Empirical/estim_kf'], 'model','results')


end



