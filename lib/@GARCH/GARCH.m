classdef GARCH < OptimProblem
  properties 
    name      = 'GARCH';
    series    = [];       			% The series used for maximum likelihood estimation
    calendar  = [];       			% (Optional) calendar associated with the above series
    cond_dist = 'Normal'; 			% The (conditional) distribution of the innovations (Normal or InverseGaussian)

    warmup_t  = 0;              % No warmup
    
    % Three values are currently dealt with:
    % 
    %   'equal'         ==> nll = -physical - risk_adjusted;
    %   'bits'          ==> nll = -physical/plen - risk_adjusted/qlen;    
    %   {'weighted',wQ} ==> nll = -(1-wQ)*physical - wQ*risk_adjusted;
    % 
    % where plen and qlen are the number of returns and options observations respectively.
    % joint_loglike = 'equal'; 
    
    % The loss function could be based on 
    %    'IVR'         ==>  Relative implied volatility 
    %    'IVA'         ==>  Absolute implied volatility 
    %    'PR'          ==>  Relative price
    %    'PA'          ==>  Absolute price
    %    'BSV'         ==>  Black-Scholes vega as done in Babaoglu et al.
    % loss_function = 'IVR'; 
    
    % Should a year count 365 days (use_busdays = false) or 252 (use_busdays = true)
    use_busdays   = true;
    
    %%#####  INTERNAL  ############################################################
    mle  = struct();   % A placeholder for MLE results
    % jmle = struct();   % A placeholder for JOINT MLE results
  
    % During the optimization, some parameter sets yield extreme variance
    % processes. For smooth optimization, set boundaries around the value of h(t).
    H_MIN  = 0.01.^2 ./ 252 ./ 3; % 1% vol
    H_MAX  = 5.0000.^2 ./ 252 ./ 3; % 500% vol
    H_FLAG = false;          % Set true if the boundaries are reached for a set of parameters   
  end % end properties

  methods 
    function self = GARCH(varargin)
    % Construct a GARCH model 
    % VARARGIN can contain pairs of the form
    %   'propname',value
    % where 'propname' must be a property defined for the GARCH model under consideration
    
      self = self@OptimProblem();
      for no = 1:2:length(varargin)
        setfield(self, varargin{no}, varargin{no+1});
      end      
      
      daily = 0.20 / sqrt(252);
      dlim  = [0.05 0.75] ./ sqrt(252);
      if self.series; daily = std(self.series); end
      self.addParameter('sigma',  daily,  dlim);        % sigma^2 is the unconditional variance (i.e. E[h])
      self.addParameter('lambda', 0.000, [-5.0, 5.00]); % ERP parameter
      self.addParameter('alpha',  0.050, [ 0.0, 0.25]);      
      self.addParameter('beta',   0.975, [ 0.8, 1.00]);
      % Note that, under the implementation of GARCH in this class, beta is
      % the persistence. If you define a subclass, in which beta is *not* the
      % persistence, it might be worth (or not) adding a (non)linear contraint so that
      % persistence < 1 in the maximum likelihood estimation of your model.
    end % GARCH
    
    %%#####  "Abstract" methods: must be implemented in subclasses  ###############

    function [loglike, z, h] = logLikelihood(self, pvalues, varargin)
    % Using PVALUES, compute the likelihood of LOG_RETURNS based on this model
      error('GARCH subclass must define logLikelihood');    
    end    

    function [exR, state] = simulateExcessReturns(self, zf, hf, dtm, shocks)      
    % Returns a matrix of (NOT LOG) *excess* returns S(t)/S(0) using Empirical
    % Martingale Simulations
    % 
    % The EMS trick is essentially (in DS1998's notation) to generate 
    %
    %   exp(-r t_j) S_i^*(t_j,n) / S_0
    %
    % rather than S_i^*(t_j,n). S_0*exp(-r t_j) should be multiplied into this
    % function's EMS output.
    %
    % ZF:     Filtered innovations (potentially useful for GARCH(P,Q) with P or Q > 1
    % HF:     Filtered variances, must be such that length(H) == length(Z)+1
    % DTM:    Simulate excess returns over the next DTM days
    % SHOCKS: TxM matrix of simulated innovations, with T >= DTM. 
      assert(length(H) == length(Z)+1)
      
      % Fetch NSIM from the dimension of the matrix
      [TMAX, nsim] = size(shocks);
      assert(dtm < TMAX)      
      
      % The initial variance is taken from the estimation part
      h = ones(dtm+1, nsim); % "dtm+1": only to make the loop simpler...
      h(1,:) = hf(end);                        
      
      error('GARCH subclass must define simulateExcessReturns');    
    end    
    
    function prices = closedFormPrices(self, options, hf, varargin)
    % Compute the prices of the OPTIONS given filtered variances HF
    % If VARARGIN is not empty, it is used to override risk-neutral parameters
      error('GARCH subclass must define closedFormPrices whenever possible');    
    end
    
    %%#####  Common methods: can be overriden in subclasses, if necessary  ########

    % DEFAULT IMPLEMENTATION: vega-weighted errors ~ N(0, var(vwe))
    function [loglike, price, errors, implied_vol] = logLikelihoodPricing(self, options, h)  
    % error('NOT EXTENSIVELY TESTED YET')
      YTM = options.YTM;
      logFK = log(options.ExDivPrice ./ options.Strike); 
      d1  = (logFK + (options.RiskFree + options.ImpliedVolOM.^2/2).*YTM) ...
            ./ (options.ImpliedVolOM.*sqrt(YTM));

      BSV = options.ExDivPrice.*normpdf(d1).*sqrt(YTM);   
      
      price  = self.closedFormPrices(options,h);

      errors = (options.OptionPrice - price)./BSV;      

      h_iv = var(errors);
      loglike = sum(normpdf(errors, 0, sqrt(h_iv)));
      if nargout > 3
        implied_vol = bmsimpv(...
            options.ExDivPrice, options.Strike, options.RiskFree, options.YTM, ...
            options.isCall, 0.0, options.ImpliedVolOM);
      end
    end % logLikelihoodPricing    

        
    function [results] = maximumLikelihood(self, options)
    % Estimates a GARCH model using maximum likelihood
      if exist('options')==1
        % Should be a lazy copy; CHECK. 
        self.mle.options = options; % For access in self.objective
      end

      
      % Do not keep the actual options (if the model is saved, you don't want to
      % save a copy of the options dataset)
      % self.mle = self.fmincon();
      self.mle = self.fminsearch(100);
      
      self.H_FLAG = false; % Reset to false
      [self.mle.loglike, self.mle.z, self.mle.h] = self.logLikelihood(self.getPValues);      
      if self.H_FLAG
        warning(['The variance process appears to hit the H_MIN or H_MAX ' ...
                 'boundary at optimum. The parameters are likely to be problematic.'])
      end

      if exist('options')==1
        self.mle.LLP = self.mle.loglike;
        [self.mle.LLQ, self.mle.prices, self.mle.errors] = self.logLikelihoodPricing(options, self.mle.h);
      end
      
      results = self.mle;
    end % maximumLikelihood

    function stderr = getStdErr(self)
      % This function computes the standard errors using the gradient
      % method coupled with returns and option data.
      T = length(self.series) - self.warmup_t;
      pv = self.getPV();
      pvalues_tmp = self.getPValues;
      
      params = self.params;
      pnames = fieldnames(params);

      free = 1;
      for pn = 1:length(pnames)
        field = pnames{pn};
        if ~params.(field).fixed
          free_pnames{free} = field;
          free = free+1;
        end
      end
      
      for fp = 1:length(free_pnames)
        for fip = 1:length(free_pnames)
          if fip ~= fp
            self.params.(free_pnames{fip}).fixed = true;
          else
            self.params.(free_pnames{fip}).fixed = false;
          end
        end
        FI = NumJacobian(@(x) -self.logLikelihood_contribution(x), pvalues_tmp(fp), self.getPValues.*0.001);
        FI = nancov(FI);
        stderr.(free_pnames{fp}) = real(sqrt(diag(pinv(FI))')./sqrt(T));

        for fip = 1:length(free_pnames)
          self.params.(free_pnames{fip}).fixed = false;
        end
        
        self.setPValues(pvalues_tmp);
      end
    end % end getStdErr

    function stderr = getStdErrOptions(self,options)
      % This function computes the standard errors using the gradient
      % method coupled with returns and option data.
      N = height(options);
      pv = self.getPV();
      pvalues_tmp = self.getPValues;

      function LL = loglike(pvalues)
        self.setPValues(pvalues);
        [~, h, q] = self.filterState();
        [~, ~, errors, ~] = self.logLikelihoodPricing(options, h, q);
        h_iv = var(errors);
        LL = log(normpdf(errors, 0, sqrt(h_iv)));
      end
      
      FI = NumJacobian(@(x) -loglike(x), self.getPValues, 0.1.*ones(size(pvalues_tmp)));
      FI = nancov(FI);
      
      if ~isnan(nanmean(mean(FI)))
        stderr_tmp = real(sqrt(diag(pinv(FI))')./sqrt(N));
      else
        stderr_tmp = NaN(size(diag(FI)))';
      end
      params = self.params;
      pnames = fieldnames(params);
      n_values = length(stderr_tmp);
      free = 0;
      for pn = 1:length(pnames)
        field = pnames{pn};
        if ~params.(field).fixed
          free = free+1;
          stderr.(field) = stderr_tmp(free);
          self.params.(field).value = pv.(field);
        end
      end
      assert(n_values == free, sprintf( ...
        'Length of pvalues (%d) is not equal to the number of free parameters (%d)', n_values, free));
      
      self.setPValues(pvalues_tmp);
    end % end getStdErrOptions
    
    % OVERRIDES OptimProblem.objective 
    function nll = objective(self, x, varargin)            
      % Under P
      [physical,z,h] = self.logLikelihood(x, varargin{:});
      plen = length(z);      
      
      nll = -physical;      
      if isfield(self.mle,'options')
        risk_adjusted = self.logLikelihoodPricing(self.mle.options,h);
        qlen = size(self.mle.options,1);
        if iscell(self.joint_loglike) && strcmp(self.joint_loglike{1}, 'weighted')
          weight = self.joint_loglike{2};
          nll = -(1-weight)*physical - weight*risk_adjusted;
        
        elseif strcmp(self.joint_loglike, 'equal')
          nll = -physical - risk_adjusted;
        
        elseif strcmp(self.joint_loglike, 'bits') 
          scale = (plen+qlen) / 2;
          nll = -scale*(physical/plen + risk_adjusted/qlen);
        
        else
          error('self.joint_loglike=''%s'' -- Not supported', self.joint_loglike);
        end
      end % if isfield(self.mle,'options')
    end % objective

    function [warmup_t] = setWarmupDate(self, date)
    % Set warmup_t to a date in the calendar; the likelihood of innovations
    % before or at this date will be ignored
      warmup_t = find(self.calendar <= date, 1, 'last');
      self.warmup_t = warmup_t;
    end % end setWarmupDate
    
  end % end methods  
end % GARCH
