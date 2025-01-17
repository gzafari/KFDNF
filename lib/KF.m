%%  Class KF
%   This class contains functions and utility to estimate the three-factor
%   model of Andersen, Fusari, Todorov 2015 by Kalman filter as proposed by Feunou and Okou 2016, KF.
classdef KF < handle
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           PROPERTIES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties
        model         = [];       % Instance of the AFT model
    end % end properties


    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           METHODS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %% Function KF: creates an instance of the KF object
        %  Input:
        %    - model:           Instance of the AFT model
        %  Output:
        %    - self:            KF object
        %
        function self = KF(model)
            self.model  = model;
        end % end KF


        %% Function logLikelihood: computes the log-likelihood and extracts
        % relevant moments of filtering distribution of latent states
        %  Output:
        %    - logLikelihood:   Log-likelihood contribution for each day
        %    - filtered:        Filtered values of latent factors
        %
        function [logLikelihood, filtered] = logLikelihood(self)
            % Number of days
            T = length(self.model.calendar);

            % Initialization
            logLikelihood    = NaN(T,1);
            filtered         = struct();

            % Copy of parameters (in a structure)
            pv = self.model.getPV();


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
            % then its log-likelihood contribution is -Inf. Also,
            % terminate the function and return a log-likelihood of -Inf
            if (sum(isnan(Gam0))~=0) || (sum(sum(isnan(Gam1)))~=0)
                logLikelihood(:) = -Inf;
                return;
            end

            % Set the variables
            % Time interval
            h           =  1/252;
            % \bar\lambda_m as defiend by Feunou and Okou 2016, p.1018
            lamd_b_m    =  2/(pv.lambdan^2);
            % \lambda_m* as defiend by Feunou and Okou 2016, p.1018
            lamd_s_m    =  24/(pv.lambdan^4);


            % Obtain the factor loadings for transition equation :
            % v_t = Phi0 + Phi1 * v_{t-1} + er
            % er~N(0,sig) ; sig = self.model.Gsig()
            % Feunou and Okou 2016, p.1018
            Phi0       = h * [pv.kappa1 * pv.nu1 + pv.mu1 * lamd_b_m * pv.cn0; pv.kappa2 * pv.nu2; pv.mu3 * lamd_b_m * pv.cn0];
            K1         = h * [-pv.kappa1 + pv.mu1 * lamd_b_m * pv.cn1, pv.mu1 * lamd_b_m * pv.cn2, pv.mu1 * lamd_b_m * pv.cn3; ...
                0, -pv.kappa2, 0; ...
                pv.mu3 * lamd_b_m * pv.cn1, pv.mu3 * lamd_b_m * pv.cn2, -pv.kappa3 + pv.mu3 * lamd_b_m * pv.cn3];

            % phi1 = I + K1
            Phi1       = eye(3) + K1;

            % obtain the determinent and inverse of K1
            detK1      = pv.kappa2 * (pv.kappa1 * pv.mu3 * lamd_b_m * pv.cn3 - pv.kappa3 * (pv.kappa1 - pv.mu1 * lamd_b_m * pv.cn1));
            K1_i       = (1/h) * (1/detK1) * [pv.kappa2 * (pv.kappa3 - pv.mu3 * lamd_b_m * pv.cn3), lamd_b_m * pv.mu1 * pv.cn2 * pv.kappa3, lamd_b_m * pv.mu1 * pv.cn3 * pv.kappa2;...
                0, pv.kappa3 * (pv.kappa1 - pv.mu1 * lamd_b_m * pv.cn1) - pv.kappa1 * pv.mu3 * lamd_b_m * pv.cn3, 0; ...
                lamd_b_m * pv.mu3 * pv.cn1 * pv.kappa2, lamd_b_m * pv.mu3 * pv.cn2 * pv.kappa1, pv.kappa2 * (pv.kappa1 - pv.mu1 * lamd_b_m * pv.cn1)];


            % Set the initial value for Vtt, and Ptt, Feunou and Okou 2016, p.1018
            x_tt      = - K1_i * Phi0;
            Px_tt     =   reshape((eye(9) - kron(Phi1,Phi1))\reshape(self.model.Gsig(x_tt, pv), [9,1]), 3, 3);


            % We iterate on each day...
            for dt = 1:T
                % We update the state based on previous state
                [loglike, x_tt, Px_tt] = self.updateState(dt, pv, x_tt, Px_tt, Phi0, Phi1, Gam0, Gam1);

                % If the current parameter set returns weights of zero,
                % then its log-likelihood contribution is -Inf. Also,
                % terminate the function and return a log-likelihood of -Inf
                if (sum(~isinf(loglike)) == 0) || (sum(~isnan(loglike)) == 0) || (sum(isreal(loglike)) == 0)
                    logLikelihood(:) = -Inf;
                    return;
                end

                % We compute filtered values
                logLikelihood(dt)      = loglike;
                filtered.mean.H1(dt)   = x_tt(1);
                filtered.mean.H2(dt)   = x_tt(2);
                filtered.mean.H3(dt)   = x_tt(3);

                % We compute standard deviation and quantiles based on asymptotic normal intervals
                filtered.std.H1(dt)    = sqrt(Px_tt(1,1));
                filtered.std.H2(dt)    = sqrt(Px_tt(2,2));
                filtered.std.H3(dt)    = sqrt(Px_tt(3,3));

                filtered.q05.H1(dt)    = filtered.mean.H1(dt) - 1.96.*filtered.std.H1(dt);
                filtered.q05.H2(dt)    = filtered.mean.H2(dt) - 1.96.*filtered.std.H2(dt);
                filtered.q05.H3(dt)    = filtered.mean.H3(dt) - 1.96.*filtered.std.H3(dt);

                filtered.q95.H1(dt)    = filtered.mean.H1(dt) + 1.96.*filtered.std.H1(dt);
                filtered.q95.H2(dt)    = filtered.mean.H2(dt) + 1.96.*filtered.std.H2(dt);
                filtered.q95.H3(dt)    = filtered.mean.H3(dt) + 1.96.*filtered.std.H3(dt);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end % end dt = 1:T
        end % end logLikelihood


        %% Function updateState: updates the state and computes the day-dt
        %  log-likelihood contribution
        %  Input:
        %    - dt:              Current day
        %    - pv:              Parameter structure
        %    - xe_old:          Mean of old state
        %    - Pe_old:          Variance of old state
        %    - phi0:            Coefficient in transition equation : v_t = Phi0 + Phi1 * v_{t-1} + er
        %    - phi1:            Coefficient in transition equation : v_t = Phi0 + Phi1 * v_{t-1} + er
        %    - Gam0:            Coefficient in cumulant equation : cumulant = Gam0 + Gam1 * V
        %    - Gam1:            Coefficient in cumulant equation : cumulant = Gam0 + Gam1 * V
        %  Output:
        %    - loglike:         Log-likelihood contribution
        %    - xtt:             Mean of the updated state
        %    - Pxtt:            Variance of the updated state
        %
        function [loglike, xtt, Ptt] = updateState(self, dt, pv, xe_old, Pe_old, Phi0, Phi1, Gam0, Gam1)
            % compute sizes of objects
            sizV                = size(self.model.moments.Variance,2);
            sizS                = size(self.model.moments.Skewness,2);
            sizK                = size(self.model.moments.Kurtosis,2);


            % and include a ridge in case something goes wrong with the variance
            Pe_old              = nearestSPD((Pe_old)) + ...
                diag([1e-11;1e-11;1e-11].*(diag(Pe_old) < [1e-11;1e-11;1e-11]));


            % update the state equation, Feunou and Okou 2016, p. 1014, Eq. 12
            xtpt       =  Phi0 + Phi1 * xe_old;
            Ptpt       =  Phi1 * Pe_old * Phi1' + self.model.Gsig(xe_old, pv);

            % get the cumulants
            Ytpt       = Gam0 + Gam1*xtpt;

            % get the observation error variances
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
            dels        = dels.^2;
            obsVec      = obsVec';

            Mtpt        = Gam1 * Ptpt * Gam1' + diag(dels);
            Mtpt        = nearestSPD(Mtpt) ;

            % and calculate the errors
            eYt         = obsVec - Ytpt;


            % update the states and make sure they are not negative
            xtt      = boxed(xtpt + Ptpt * Gam1'*(Mtpt\eYt), self.model.H_MIN, self.model.H_MAX);


            % update the state variane
            Ptt      =  Ptpt - Ptpt * Gam1' * (Mtpt\(Gam1 * Ptpt));
            Ptt      =  nearestSPD(Ptt) + diag([1e-11;1e-11;1e-11].*(diag(Ptt) < [1e-11;1e-11;1e-11]));

            % and calculate the log-likelihood
            loglike  = -0.5 * (size(eYt,1) * log(2*pi) + log(det(Mtpt)) + eYt' * (Mtpt\eYt));


        end % end updateState


    end % end methods

end % end classdef KF