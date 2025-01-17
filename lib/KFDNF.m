%%  Class KFDNF
%   This class contains functions and utility to estimate the three-factor
%   model of Andersen, Fusari, Todorov 2015 with the method proposed in Bégin and Zafari (2025).
%   The estimation method is based on a combination of the Kalman filter (KF) and the discrete nonlinear filter (DNF), KFDNF.
classdef KFDNF < handle
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           PROPERTIES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties
        K             = [];             % Grid size for DNF
        model         = [];             % Instance of the AFT model
    end % end properties

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           METHODS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        %% Function KFDNF: creates an instance of the KFNDF object
        %  Input:
        %    - model:           Instance of the AFT model
        %    - K:               Jump grid size nodes
        %  Output:
        %    - self:            KFDNF object
        %
        function self = KFDNF(model, K)
            self.model  = model;

            if nargin > 1
                self.K   = K;
            else
                % set a default value for K
                self.K   = 5;
            end
        end % end KFDNF


        %% Function createDNFGrid: creates DNF grid as the modified version of Bégin and Boudreault(2020).
        %  Input:
        %    - pv:              Parameter structure (from getPV)
        %  Output:
        %    - grid:            Jump grid
        %
        function grid = createDNFGrid(self, pv)
            % The modified deltaK makes the grid more concenterated around the mean.
            deltaK          = 3 * log(self.K);%/2;

            % Mean and variance of jump components (the Poisson arrival
            % intensity is not considered in the mean and variance calculation as
            % we are conditioning on the existance of jumps).
            EXN             = - 1/pv.lambdan;
            EY              = - 1/pv.lambdan;

            VXN             = - EXN./pv.lambdan;
            VY              = - EY ./pv.lambdan;

            % As there are negative jumps, the truncatatioins are at zero upper bound
            grid.chiXN      = (EXN - deltaK*sqrt(VXN)).* (0:1:self.K)./self.K;
            grid.chiY       = (EY  - deltaK*sqrt(VY)) .* (0:1:self.K)./self.K;

            if self.K == 0
                grid.chiXN  = [0];
                grid.chiY   = [0];
            end
        end % end createDNFGrid



        %% Function logLikelihood: computes the log-likelihood and extracts
        % relevant moments of filtering distribution of latent states
        %  Output:
        %    - logLikelihood:   Log-likelihood contribution for each day
        %    - filtered:        Filtered values of latent factors
        %
        function [logLikelihood, filtered] = logLikelihood(self)
            % Number of days
            T    = length(self.model.calendar);

            % Time interval
            h    = 1/252;

            % Copy of parameters (in a structure)
            pv   = self.model.getPV();

            % Initialization
            logLikelihood   = NaN(T,1);
            filtered        = struct();


            % If the current parameter does not satisfies the constraints,
            % then its log-likelihood contribution is -Inf. Also,
            % terminate the function and return a log-likelihood of -Inf
            if (pv.kappa2 <= 0) || (pv.kappa1 <= (2*pv.mu1*pv.cn1)/pv.lambdan^2) || (pv.kappa3 <= (2*pv.mu3*pv.cn3*pv.kappa1)/(pv.kappa1*pv.lambdan^2 - 2*pv.mu1*pv.cn1)) || (pv.sigma1^2 > 2*pv.kappa1*pv.nu1) || (pv.sigma2^2 > 2*pv.kappa2*pv.nu2)
                logLikelihood(:) = -Inf;
                return;
            end

            % Obtain Gam0 and Gam1 such that cumulant = Gam0 + Gam1 * V
            [Gam0, Gam1]    = self.model.getCumulants();

            % If the current parameter generates NaN in Gam0 or Gam1,
            % then its log-likelihood contribution is -Inf. Also
            % terminate the function and return a log-likelihood of -Inf
            if (sum(isnan(Gam0))~=0) || (sum(sum(isnan(Gam1)))~=0)
                logLikelihood(:) = -Inf;
                return;
            end


            % create the jump grid as well as the mid-nodes
            grid             = self.createDNFGrid(pv);
            grid.chiMidXN    = [0, (grid.chiXN(2:end-1) + grid.chiXN(3:end))/2,  -Inf];
            grid.chiMidY     = [0, (grid.chiY(2:end-1)  + grid.chiY(3:end) )/2,  -Inf];


            % obtain the factor loadings for transition equation :
            % v_t = Phi0 + Phi1 * v_{t-1} + er
            % er~N(0,sig) ; sig = self.jCondGsig()
            % whenever applicable, last two dimensions accounts for xn and y, respectively.
            Phi0                = zeros(3,1,self.K+1,self.K+1);
            Phi0(1,1,:,:)       = repmat(pv.kappa1.* pv.nu1.* h + pv.mu1.*reshape(grid.chiXN.^2 , 1, 1,self.K+1),1,1,1,self.K+1);
            Phi0(2,1,:,:)       = repmat(pv.kappa2.* pv.nu2.* h,1,1,self.K+1,self.K+1);
            Phi0(3,1,:,:)       = pv.mu3.*( (1-pv.rho3).*reshape(grid.chiXN.^2 , 1 , 1 , self.K+1) ...
                + pv.rho3.*reshape(grid.chiY.^2 , 1 , 1 , 1 , self.K+1));

            % phi1 = I + K1
            K1       = [-pv.kappa1 * h,0,0; 0, -pv.kappa2*h, 0; 0,0, -pv.kappa3*h];
            Phi1     = repmat(eye(3) + K1,1,1,self.K+1,self.K+1);


            % set initial state to initial values (parameter values) and
            % initial variance to zero (as initial values are assumed as
            % parameters in the system)
            x_tt            = [pv.v10;pv.v20;pv.v30];
            Px_tt           = 0*diag([1,1,1]);


            % iterate on each day...
            for dt = 1:T

                % update the state based on previous state
                [loglike, x_tt, Px_tt] = self.updateState(dt, pv, x_tt, Px_tt, grid, Phi0, Phi1, Gam0, Gam1);

                % If the current parameter set returns weights of zero,
                % then its log-likelihood contribution is -Inf. Also,
                % terminate the function and return a log-likelihood of -Inf
                if (sum(~isinf(loglike)) == 0) || (sum(~isnan(loglike)) == 0) || (sum(isreal(loglike)) == 0)
                    logLikelihood(:) = -Inf;
                    return;
                end

                % compute filtered values
                logLikelihood(dt)      = loglike;
                filtered.mean.H1(dt)   = x_tt(1);
                filtered.mean.H2(dt)   = x_tt(2);
                filtered.mean.H3(dt)   = x_tt(3);


                % compute standard deviation and quantiles based on asymptotic normal intervals
                filtered.std.H1(dt)    = sqrt(Px_tt(1,1));
                filtered.std.H2(dt)    = sqrt(Px_tt(2,2));
                filtered.std.H3(dt)    = sqrt(Px_tt(3,3));

                filtered.q05.H1(dt)    = filtered.mean.H1(dt) - 1.96.*filtered.std.H1(dt);
                filtered.q05.H2(dt)    = filtered.mean.H2(dt) - 1.96.*filtered.std.H2(dt);
                filtered.q05.H3(dt)    = filtered.mean.H3(dt) - 1.96.*filtered.std.H3(dt);

                filtered.q95.H1(dt)    = filtered.mean.H1(dt) + 1.96.*filtered.std.H1(dt);
                filtered.q95.H2(dt)    = filtered.mean.H2(dt) + 1.96.*filtered.std.H2(dt);
                filtered.q95.H3(dt)    = filtered.mean.H3(dt) + 1.96.*filtered.std.H3(dt);
            end % end dt = 1:T
        end % end logLikelihood



        %% Function jCondGsig: computes the variance of noise in transition equation condition on having jumps
        %  Input:
        %    - V:         Value of factors
        %    - pv:        Parameter structure
        %  Output:
        %    - sig:       Noise variance of transition equation
        %
        function [sig] = jCondGsig(self, V, pv)
            h          =  1/252;
            sig        =  zeros(3,3);
            sig(1,1)   =  pv.sigma1^2 * V(1)*h;
            sig(2,2)   =  pv.sigma2^2 * V(2)*h;
            sig(3,3)   =  eps;
        end % end jCondGsig


        %% Function updateState: updates the state and computes the day-dt
        %  log-likelihood contribution
        %  Input:
        %    - dt:              Current day
        %    - pv:              Parameter structure
        %    - xe_old:          Mean of old state
        %    - Pe_old:          Variance of old state
        %    - grid:            Jump grid
        %    - phi0:            Coefficient in transition equation : v_t = Phi0 + Phi1 * v_{t-1} + er
        %    - phi1:            Coefficient in transition equation : v_t = Phi0 + Phi1 * v_{t-1} + er
        %    - Gam0:            Coefficient in cumulant equation : cumulant = Gam0 + Gam1 * V
        %    - Gam1:            Coefficient in cumulant equation : cumulant = Gam0 + Gam1 * V
        %  Output:
        %    - loglike:         Log-likelihood contribution
        %    - x_tt:            Mean of the updated state
        %    - Px_tt:           Variance of the updated state
        %
        function [loglike, x_tt, Px_tt] = updateState(self, dt, pv, xe_old, Pe_old, grid, Phi0, Phi1, Gam0, Gam1)
            % compute sizes of objects
            sizV                = size(self.model.moments.Variance,2);
            sizS                = size(self.model.moments.Skewness,2);
            sizK                = size(self.model.moments.Kurtosis,2);


            % include a ridge in case something goes wrong with the variance
            Pe_old              = nearestSPD((Pe_old)) + ...
                diag([[1e-11;1e-11;1e-11].*(diag(Pe_old) < [1e-11;1e-11;1e-11])]);


            % update the state equation
            xtpt               = Phi0 + repmat(squeeze(Phi1(:,:,1,1)) * xe_old, 1, 1, self.K+1, self.K+1);

            % and its variance (this is a 2-d matrix as it does not depend on jump grids, i.e., Phi1 does not depend on the jumps)
            Ptpt2d             = squeeze(Phi1(:,:,1,1)) * Pe_old * squeeze(Phi1(:,:,1,1))' + self.jCondGsig(xe_old, pv);

            % get the cumulants
            y_tpt              = repmat(Gam0, 1, 1, self.K+1, self.K+1) + reshape(Gam1 * reshape(xtpt, 3, []), [], 1, self.K+1, self.K+1);


            % compute jump probabilities
            [~,pXN,pY]        = self.getJumpProbabilities(pv, xe_old, grid);


            % obtain the observation error variances
            if sizV > 0
                dels      = [pv.del2.*std(self.model.moments.Variance)];
                obsVec    = [self.model.moments.Variance(dt,:)];
            end
            if sizS > 0
                dels      = [dels, pv.del3.*std(self.model.moments.Skewness.*self.model.moments.Variance.^1.5)];
                obsVec    = [obsVec,self.model.moments.Skewness(dt,:).*self.model.moments.Variance(dt,:).^1.5];
            end
            if sizK > 0
                dels      = [dels, pv.del4.*std((self.model.moments.Kurtosis-3).*self.model.moments.Variance.^2)];
                obsVec    = [obsVec,(self.model.moments.Kurtosis(dt,:)-3).*self.model.moments.Variance(dt,:).^2];
            end
            dels          = dels.^2;
            obsVec        = obsVec';

            Mtpt2d        = Gam1 * Ptpt2d *Gam1' + diag(dels);
            P_yy          = nearestSPD(Mtpt2d) ;


            % update the state variance
            P_tt          = Ptpt2d - Ptpt2d * Gam1' * (P_yy\(Gam1 * Ptpt2d));
            P_tt          = nearestSPD(P_tt) + diag([1e-11;1e-11;1e-11].*(diag(P_tt) < [1e-11;1e-11;1e-11]));

            % and obtain the log-likelihood and covariance matrices by aggregating the various
            % outputs from the KF updating
            for dxn = 1:((self.K+1)*(length(grid.chiXN)>1) + (length(grid.chiXN)==1))
                for dy = 1:((self.K+1)*(length(grid.chiY)>1) + (length(grid.chiY)==1))

                    loglikes(dxn,dy)   = - 0.5 * ((length(y_tpt(:,:,dxn,dy))) * log((2*pi)) + log(det(P_yy)) + ((obsVec - y_tpt(:,:,dxn,dy))') * (P_yy\(obsVec - y_tpt(:,:,dxn,dy))));
                    x_tts(:,dxn,dy)    =   boxed(xtpt(:,:,dxn,dy) + (Ptpt2d*Gam1') * (P_yy\(obsVec - y_tpt(:,:,dxn,dy))), self.model.H_MIN, self.model.H_MAX);

                    try
                        x2_tts(:,:,dxn,dy) =  nearestSPD(P_tt + x_tts(:,dxn,dy)*x_tts(:,dxn,dy)');
                    catch
                        x_tt    = NaN;
                        Px_tt   = NaN;
                        loglike = NaN;
                        x2_tts(:,:,dxn,dy) = NaN;
                    end

                end
            end

            % reshape the jump probabilities
            Ps        = reshape(pXN,self.K+1,1).*reshape(pY,1,self.K+1);

            % calculate the likelihood by aggregating the likelihood for
            % different node in the jump grid
            mLs       = max(loglikes(:));
            Ls        = exp(loglikes-mLs).*Ps;
            like      = sum(Ls,'all');

            % aggregate the states as the wighted average of states
            % curresponding to different nodes in the grid
            x_tt      = sum(x_tts .*repmat(reshape(Ls./like,[1,self.K+1,self.K+1]),3,1,1),[2,3]);
            x_tt      = boxed(x_tt, self.model.H_MIN, self.model.H_MAX);
            x2_tt     = sum(x2_tts.*repmat(reshape(Ls./like,[1,1,self.K+1,self.K+1]),3,3,1,1),[3,4]);


            % obtain the loh-likelihood and the states variance-covariance matrix
            loglike   = log(like) + mLs;
            try
                Px_tt     = nearestSPD(x2_tt - x_tt*x_tt')+diag([1e-11;1e-11;1e-11].*(diag(x2_tt - x_tt*x_tt') < [1e-11;1e-11;1e-11]));
            catch
                x_tt    = NaN;
                Px_tt   = NaN;
                loglike = NaN;
            end

        end % end updateState


        %% Function getJumpProbabilities: calculates the jump probabilities
        %  Input:
        %    - pv:              Parameter structure
        %    - chiX:            Variance factors
        %    - grid:            Jump grid
        %  Output:
        %    - pXP:             Probability of positive jump
        %    - pXN:             Probability of negative jump
        %    - pY:              Probability of independent negative jump
        %
        function [pXP,pXN,pY] = getJumpProbabilities(self, pv, chiX, grid)
            % calculate the jump intensities
            delta_t    = 1/252;
            [HP, HN]   = self.model.getJumpIntensity(pv, chiX(1,:), chiX(2,:), chiX(3,:));
            HP         = HP * delta_t;
            HN         = HN * delta_t;
            HY         = HN;

            % probability of positive jump (not used if return is not included in the observations, as is the case in this work)
            pXP            = NaN(length(HP),self.K+1);

            % probability of negative jump
            pXN            = NaN(length(HN),self.K+1);
            for dxn = 1:self.K
                if grid.chiXN(dxn) == 0
                    pXN(:,dxn) = exp(-HN);
                else
                    pXN(:,dxn) = (1-exp(-HN)) ...
                        .*(-exp(grid.chiMidXN(dxn)*pv.lambdan) + exp(grid.chiMidXN(dxn-1)*pv.lambdan));
                end
            end
            if self.K > 0
                pXN(:,self.K+1) = 1 - sum(pXN(:,1:self.K),2);
            else
                pXN(:,self.K+1) = 1;
            end


            % probability of negative independent jump
            pY             = NaN(length(HY),self.K+1);
            for dy = 1:self.K
                if grid.chiY(dy) == 0
                    pY(:,dy) = exp(-HY);
                else
                    pY(:,dy) = (1-exp(-HY)) ...
                        .*(-exp(grid.chiMidY(dy)*pv.lambdan) + exp(grid.chiMidY(dy-1)*pv.lambdan));
                end
            end
            if self.K > 0
                pY(:,self.K+1)  = 1 - sum(pY(:,1:self.K),2);
            else
                pY(:,self.K+1)  = 1;
            end
        end % end getJumpProbabilities

    end % end methods

end % end classdef KFDNF