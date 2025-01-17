% This class is a base class for any model dealing with (potentially)
% constrained parameters
% 
% The subclass must define the "objective" method. It may redefine the
% "linearConstraints" and "nonlinearConstraints" methods. 
%
% TODO: It could be interesting if this class was somewhat consistent with
% GlobalOptimSolution or had a GlobalOptimSolution property. Unfortunately,
% MultiStart does seem to support fminsearch, but this could easily be
% implemented here (in a parfor at that) assuming that proper deep-copy is
% implemented in OptimProblem and subclasses and in the Param class.
classdef OptimProblem < handle
  properties 
    params = struct();
    optimopt = [];
    
    % The getPValues and setPValues are typically used to interact with an
    % optimizer. Sometimes, it can be helpful to scale the parameters before
    % optimization (and after, obviously) so as to put all parameters on
    % comparable scales. The default behavior is NOT to scale parameters. If
    % scale_pvalues is true, the BOUNDS properties of all parameters must be set
    % to finite values that reasonnably box the expected optimal values. These
    % BOUNDS will be used to scale the parameter and optimization will become
    % sensitive to the value of these BOUNDS...
    scale_pvalues = false;
  end % end properties

  methods(Static)
    function stderr = bwerr(grad,hessian)
      B = grad * grad';
      Ainv = pinv(-hessian); % The pseudo-inverse of the information matrix
      stderr = sqrt( diag(Ainv * B * Ainv) );
    end
  end % methods(Static)

  
  methods 
    function self = OptimProblem(varargin)
      self.addParameter(varargin{:});
      
      options = optimset('fmincon');
      options.Algorithm = 'interior-point';
      options.MaxIter   = 250;
      options.TolFun    = 1e-4;
      options.TolX      = 1e-4;
      options.Display   = 'iter';

      self.optimopt = options;
    end % OptimProblem      

    
    %%#####  Problem definition  ##################################################
    function value = objective(self, x, varargin)
      error('Subclass must define the objective function');
    end % objective
    
    function [A,b,Aeq,beq] = linearConstraints(self)
    % Returns the linear constraints on the fmincom problem. By default,
    % there are no constraints.
      A = [];  b = [];  Aeq = [];  beq = [];
    end % linearConstraints
    

    function [c,ceq] = nonlinearConstraints(self, x)
    % Returns the nonlinear constraints on the fmincom problem. By default,
    % there are no constraints.
      c = [];  ceq = [];
    end % nonlinearConstraints   
    %%#####  END Problem definition  ##############################################

    
    %%#####  Define and manipulate parameters  ####################################
    function self = addParameter(self, varargin)
    % VARARGIN must be a sequence of triplets: name, value, bounds
      N = length(varargin);
      assert( mod(N,3) == 0 );
      for no = 1:3:N
        self.params.(varargin{no}) = Param(varargin{no}, varargin{no+1}, varargin{no+2}); 
      end
    end % addParameter

    
    function self = removeParameter(self, varargin)
    % VARARGIN must be a list of parameter names
      N = length(varargin);
      for no = 1:N
        self.params = rmfield(self.params, varargin{no});
      end
    end % removeParameter
    

    function [] = fetchParameters(self)
    % This function assigns the parameters as variables in the caller
    % environment. This is here for convenience, but its use is **not
    % recommended**. Indeed, if the parameter name (e.g. beta) is the same as any matlab
    % function, the function might not be overriden! It is better to use,
    % e.g.
    %     pv = problem.getPV();
    %     beta2 = pv.beta.^2;
      params = self.params;
      pnames = fieldnames(params);
        
      for fno = 1:length(pnames)
        field = pnames{fno};
        value = params.(field).value;
        assignin('caller', field, value);
      end  
    end % fetchParameters

      
    function pv = getPV(self)
    % Return parameters as a structure with field named after the parameters,
    % and with their values set to the parameter values
      pv = self.params;
      pnames = fieldnames(pv); 
      for fno = 1:length(pnames)
        field = pnames{fno};
        pv.(field) = pv.(field).value;
      end  
    end % getPV
      

    function pv = getFreePV(self)
    % Return unconstrained parameters as a structure with field named after the parameters,
    % and with their values set to the parameter values
    % NOTE THAT implementing this function without calling getPV would be
    % dangerous! Indeed, if getPV is overriden but not getFreePV, results can
    % be inconsistent...
      pv = self.getPV();
      pnames = fieldnames(pv);       
      for fno = 1:length(pnames)
        field = pnames{fno};
        if ~isfield(self.params,field) || self.params.(field).fixed
          pv = rmfield(pv,field);
        end
      end  
    end % getFreePV

    
    function [pvalues,bounds,names] = getPValues(self)
    % Returns a vector of numeric values for the *free* parameters
    % Can also return the associated bounds and names. The names of the free
    % parameters can be useful do build constraints on parameters.
      params = self.params;
      pnames = fieldnames(params);
        
      pvalues = [];
      bounds = [];
      names = {};
      for fno = 1:length(pnames)
        field = pnames{fno};
        if ~params.(field).fixed
          if self.scale_pvalues
            ISM = params.(field).inverse_sig_max;
            pvalues(end+1) = params.(field).getScaledValue();
            bounds(end+1,:) = [-ISM, ISM]; 
            if abs(abs(pvalues(end))-ISM) < 1e-6; 
              warning('Scaled pvalue for parameter %s hits +/- ISM', field);
            end
          else
            pvalues(end+1) = params.(field).value;
            bounds(end+1,:) = params.(field).bounds;
          end
          names{end+1} = field;
        end
      end  
    end % getPValues            

    
    function [A] = getConstraintRow(self, varargin)
    % Returns a row suitable for the A or Aeq matrix in fmincon
    % varargin must contain "coeff,pname" pairs
      nargin = length(varargin); % Don't count self
      assert(mod(nargin,2) == 0, 'Must provide "coeff,pname" pairs')
      [~,~,names] = self.getPValues();
      N = length(names);

      A = zeros(1,N);
      for no = 1:2:nargin
        coeff = varargin{no};
        sx = strcmp(names, varargin{no+1});
        assert( sum(sx)==1 );
        A(sx) = coeff;
      end
    end

    
    function [values] = getParameterValues(self,varargin)
    % Return the value of the parameters named in VARARGIN
      N = length(varargin);
      params = self.params;
      values = NaN(size(varargin));
      for pn = 1:N        
        values(pn) = params.(varargin{pn}).value;
      end  
    end % getParameterValues      

    
    function [pv] = setPV(self,pv)
      params = self.params;     
      pnames = fieldnames(pv);
      assert(all(ismember(pnames,fieldnames(params))), 'Setting value of invalid parameter')      
      for pn = 1:length(pnames)
        field = pnames{pn};
        params.(field).value = pv.(field);
      end
      self.params = params;      
      pv = self.getPV(); % For convenience
    end
      
    function [] = setParameterValues(self,varargin)
    % Set the value of the parameters named in VARARGIN; must consist of
    % "pname,value" pairs.
      params = self.params;     
      pnames = varargin(1:2:end);
      values = varargin(2:2:end);
      assert(all(ismember(pnames,fieldnames(params))), 'Setting value of invalid parameter')
      
      for pn = 1:length(pnames)
        field = pnames{pn};
        params.(field).value = values{pn};
      end  
      self.params = params;
    end % setParameterValues      

      
    function [] = setPValues(self,pvalues)
    % Set the value of the *free* parameters. The length of the PVALUES
    % vector must be equal to the number of free parameters.
      params = self.params;
      pnames = fieldnames(params);      
      n_values = length(pvalues);
      
      free = 0;
      for pn = 1:length(pnames)
        field = pnames{pn};
        if ~params.(field).fixed
          free = free+1;
          if self.scale_pvalues
            params.(field).setScaledValue( pvalues(free) );
          else          
            params.(field).value = pvalues(free);
          end
        end
      end  
      assert(n_values == free, sprintf(...
          'Length of pvalues (%d) is not equal to the number of free parameters (%d)',...
          n_values, free));
    end % setPValues       
    %%#####  END Define and manipulate parameters  ################################
    
    
    %%#####  Actual optimization  #################################################
    
    function [] = finalizeOptim(self, results, varargin)
      % Default: do nothing 
    end    
    
    function [results] = fmincon(self, varargin)
    % Estimates a model using maximum likelihood. VARARGINs are forwarded to
    % the objective method
      [A,b,Aeq,beq] = self.linearConstraints();
      nlcon = @(x) self.nonlinearConstraints(x);
      
      tic;
      [x0,bounds] = self.getPValues();
      disp( dataset({[bounds(:,1), x0(:), bounds(:,2)], 'LB','PV0', 'UB'}) )
      [params,nll,eflag,output,lambda,grad,hessian] = fmincon(...
          @(x)self.objective(x, varargin{:}), x0, ...
          A,b,Aeq,beq, bounds(:,1)'-eps,bounds(:,2)'+eps, nlcon, self.optimopt);
      self.setPValues(params);      
      optim_time = toc;
            
      results = self.getPV();
      results.nll = nll;
      results.x0  = x0;
      results.grad = grad;
      results.hessian = hessian;
      results.stderr = self.bwerr(grad,hessian);
      results.optim.eflag = eflag;
      results.optim.output = output;
      results.optim.lambda = lambda;
      results.optim.time = optim_time;
      %JF: self.getOutputs(method_obj); ??
      
      if eflag == -2
        self.setPValues(x0);
        warning(['Param values out of optimizer neglected as the optimization ' ...
                 'terminated at an unfeasible point. Reverting the x0.']);
      end
      
      if self.scale_pvalues
        warning('OptimProblem:fmincon', 'grad and hessian are on the scaled parameters')
      end
      self.finalizeOptim(results, varargin{:});
    end % fmincon

    
    function [results] = fminsearch(self, lambda, varargin)
    % Dirty way to transform the constrained problem in an unconstrained
    % problem: penalize the constraints. See penalizeConstraints. VARARGINs are forwarded to
    % the objective method
      optimopt = optimset('fminsearch');
      optimopt.MaxIter = self.optimopt.MaxIter;
      optimopt.MaxFunEvals = self.optimopt.MaxIter;
      optimopt.Display = self.optimopt.Display;
      optimopt.TolFun  = self.optimopt.TolFun;
      optimopt.UseParallel = self.optimopt.UseParallel;
      optimopt.TolX = self.optimopt.TolX;
      
      tic;      
      [pvalues,bounds] = self.getPValues();
      disp( dataset({[bounds(:,1), pvalues(:), bounds(:,2)], 'LB','PV0', 'UB'}) )
      objective = @(x) self.penalizeConstraints(x,bounds,lambda,varargin{:});
      [params,fmin] = fminsearch(objective, pvalues, optimopt);      
      [fval,params] = objective(params); % Apply lb and ub!
      self.setPValues(params);      
      assert(abs(fmin-fval)<0.001)
      optim_time = toc;      
      
      results = self.getPV();
      results.fmin = fmin;
      results.optim_time = optim_time;
      self.finalizeOptim(results, varargin{:});
    end % end fminsearch
    
    function [results] = patternsearch(self, varargin)
    % Estimates a model using maximum likelihood. VARARGINs are forwarded to
    % the objective method
      optimopt         = optimoptions('patternsearch');
      optimopt.Display = self.optimopt.Display;

      [A,b,Aeq,beq] = self.linearConstraints();
      nlcon = @(x) self.nonlinearConstraints(x);
      
      tic;
      [x0,bounds] = self.getPValues();
      disp( dataset({[bounds(:,1), x0(:), bounds(:,2)], 'LB','PV0', 'UB'}) )
      [params,nll,eflag,output] = patternsearch(...
          @(x)self.objective(x, varargin{:}), x0, ...
          A,b,Aeq,beq, bounds(:,1)'-eps,bounds(:,2)'+eps, nlcon, optimopt);
      self.setPValues(params);
      optim_time = toc;
            
      results = self.getPV();
      results.nll = nll;
      results.x0  = x0;
      results.optim.eflag = eflag;
      results.optim.output = output;
      results.optim.time = optim_time;
      
      if eflag == -2
        self.setPValues(x0);
        warning(['Param values out of optimizer neglected as the optimization ' ...
                 'terminated at an unfeasible point. Reverting the x0.']);
      end
      
      if self.scale_pvalues
        warning('OptimProblem:fmincon', 'grad and hessian are on the scaled parameters')
      end
      self.finalizeOptim(results, varargin{:});
    end % patternsearch
       
    
    function [penalized,x,cstr] = penalizeConstraints(self,x,bounds,lambda,varargin)      
    % Dirty way to transform the constrained problem in an unconstrained
    % problem: penalize the constraints...
      x = x(:);
      cstr = [];
      xc = min(max(x(:),bounds(:,1)), bounds(:,2));
      penalized = max(abs(x - xc));
      if penalized > 0
        cstr(end+1) = penalized;
      end
      x = xc;
      
      obj = self.objective(x, varargin{:});      
      [A,b,Aeq,beq] = self.linearConstraints();
      [c,ceq] = self.nonlinearConstraints(x);
      
      if A
        penalized = max([A*x - b; penalized]);
        cstr(end+1) = penalized;
      end
      if Aeq
        penalized = max([abs(Aeq*x - beq); penalized]);
        cstr(end+1) = penalized;
      end
      if c
        penalized = max([c(:); penalized]);
        cstr(end+1) = penalized;
      end
      if ceq
        penalized = max([abs(ceq(:)); penalized]);
        cstr(end+1) = penalized;
      end
      assert(penalized >= 0);
      
      penalized = obj + lambda*penalized;
    end % penalizeConstraints
          

    % OUTDATED AND NOT TESTED
    % function [results] = fminGlobalSearch(self)
    % % Estimates a model using maximum likelihood
    %   [A,b,Aeq,beq] = self.linearConstraints();
    %   nlcon = @(x) self.nonlinearConstraints(x);
    %   
    %   [pvalues,bounds] = self.getPValues();
    %   disp( dataset({[bounds(:,1), pvalues(:), bounds(:,2)], 'LB','PV0', 'UB'}) )
    %   problem = createOptimProblem(...
    %       'fmincon','objective',@(x)self.objective(x),...
    %       'x0',pvalues, 'lb',bounds(:,1)'-eps,'ub',bounds(:,2)'+eps,...
    %       'Aineq',A,'bineq',b,'Aeq',Aeq,'beq',beq, 'nonlcon',nlcon, 'options',self.optimopt);
    %   % ms = MultiStart('StartPointsToRun','bounds');
    %   gs = GlobalSearch('StartPointsToRun','bounds');
    %   [params,fmin,eflag,output,solutions] = run(gs,problem);      
    %   self.setPValues(params);      
    %         
    %   results = self.getPV();
    %   results.fmin = fmin;
    %   results.eflag = eflag;
    %   results.output = output;
    %   results.solutions = solutions;
    % end % fminGlobalSearch
    
    function []=illustrate_scaling(self,name,initial_parameters,bounds,final_parameters)  
        
        ISM = 5;
        y = -ISM:0.1:ISM;
        figure('units','normalized','outerposition',[0 0.05 .9 .9],'Color',[1 1 1]);
        
        j=1;%incrément pour les subplot
        for i = 1:length(initial_parameters)
                subplot(length(initial_parameters),2,j);hold on
                lb = bounds(i,1);   
                ub = bounds(i,2);
                z = Param.unscale(y,lb,ub);
                plot(y, z,'r')
                hline(initial_parameters(i),'k-');
                hline(final_parameters(i),'k--');
                xlabel('Optim value'); 
                ylabel({'Param value'; sprintf('[%d, %d]',lb,ub)});
                str = name{i};
                title(str);
%                DIFF = [z(1)+ub, z(end)+lb] ./ Mx
                j=j+1;
                subplot(length(initial_parameters),2,j);hold on
                plot(z,y,'r')
                vline(initial_parameters(i),'k-');
                vline(final_parameters(i),'k--');
                xlabel('Param value'); 
                ylabel({'Optim value'; sprintf('[-%d, %d]',-ISM,ISM)});
                title(str)
%                 DIFF = [z(1)+ub, z(end)+lb] ./ Mx
                j=j+1;
        end
        
    end
  end % methods
end
