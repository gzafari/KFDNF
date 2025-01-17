%%  Class ParticleFilterSIR
%   This class contains functions and utility to estimate the three-factor
%   model of Andersen, Fusari, Todorov 2015 by the SIR particle filter
classdef ParticleFilterSIR < handle
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           PROPERTIES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties
        nb_particles  = 10000;           % 10000 particles by default
        nb_partitions = 5;               % Number of partitions for Malik and Pitt's method
        resampling    = 'Stratified';    % Resampling method
        model         = [];              % Instance of the AFT model
        particles     = [];              % Values of the particles used to compute the likelihood
    end % end properties

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           METHODS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %% Function ParticleFilterSIR: constructs the ParticleFilterSIR object
        %  Input:
        %    - model:           Instance of the aft model
        %    - nb_particles:    Number of particles
        %    - resampling:      Resampling method
        %  Output:
        %    - self:            ParticleFilterSIR object
        %
        function self = ParticleFilterSIR(model, nb_particles, resampling)
            if nargin > 1
                self.nb_particles = nb_particles;
                self.resampling   = resampling;
            end
            self.model = model;
            self.generateParticles();
        end % end ParticleFilterSIR

        %% Function generateParticles: generates particles
        %
        function [] = generateParticles(self)
            % compute how many particles are needed (and we cap them at 1000
            % days, assuming we reuse after that)
            T                           = min(1000,length(self.model.series));

            % generate the particles; note that we assume that there is at
            % most one jump per day (per type of jump)
            self.particles.psi1         = normrnd(0,1,self.nb_particles,T);
            self.particles.psi2         = normrnd(0,1,self.nb_particles,T);
            self.particles.psi3         = normrnd(0,1,self.nb_particles,T);
            self.particles.np           = unifrnd(0,1,self.nb_particles,T);
            self.particles.nn           = unifrnd(0,1,self.nb_particles,T);
            self.particles.ny           = unifrnd(0,1,self.nb_particles,T);
            self.particles.xp           = unifrnd(0,1,self.nb_particles,T);
            self.particles.xn           = unifrnd(0,1,self.nb_particles,T);
            self.particles.xy           = unifrnd(0,1,self.nb_particles,T);
            self.particles.unif_resam   = unifrnd(0,1,3,T);
        end % end generateParticles

        %% Function logLikelihood: obtains log-likelihood and filtered values
        %  Output:
        %    - logLikelihood:   Log-likelihood contribution for each day
        %    - filtered:        Filtered values of latent factors
        function [logLikelihood, filtered] = logLikelihood(self)
            % This function applies the SIR methodology to obtain the
            % likelihood function and the various filtered values.
            T               = length(self.model.series);
            logLikelihood   = NaN(T,1);
            filtered        = struct();

            % Copy of parameter structure
            pv              = self.model.getPV();

            % Initialization
            H1              = pv.v10.*ones(self.nb_particles,1);
            H2              = pv.v20.*ones(self.nb_particles,1);
            H3              = pv.v30.*ones(self.nb_particles,1);

            % For each day...
            for dt = 1:T
                dt
                [weights, weights_loglike, H1, H2, H3, XP, XN, Y] = self.updateState(dt, pv, H1, H2, H3);
                normalizedweights = weights./sum(weights);

                % If the current parameter set returns weights of zero for every
                % particle, then its log-likelihood contribution is -Inf. Also,
                % terminate the program.
                if sum(~isnan(normalizedweights)) == 0
                    logLikelihood(:) = -Inf;
                    return;
                end

                % compute filtered values based on normalized weights
                logLikelihood(dt)      = log(mean(exp(weights_loglike - max(weights_loglike)))) + max(weights_loglike);
                filtered.mean.H1(dt)   = H1'*normalizedweights;
                filtered.mean.H2(dt)   = H2'*normalizedweights;
                filtered.mean.H3(dt)   = H3'*normalizedweights;
                filtered.mean.XP(dt)   = XP'*normalizedweights;
                filtered.mean.XN(dt)   = XN'*normalizedweights;
                filtered.mean.Y(dt)    = Y'*normalizedweights;

                % resampling; one method is implemented so far.
                switch self.resampling
                    case 'Stratified'
                        I             = randsample_stratified_cpp(weights, self.particles.unif_resam(1,mod(dt-1,1000)+1));
                        H1            = H1(I);
                        H2            = H2(I);
                        H3            = H3(I);
                    otherwise
                        error('Specified resampling method not implemented.');
                end % end switch self.resampling

                % compute standard deviation and quantiles based on resampled sample
                filtered.std.H1(dt)    = std(H1);
                filtered.std.H2(dt)    = std(H2);
                filtered.std.H3(dt)    = std(H3);

                filtered.q05.H1(dt)    = quantile(H1,0.025);
                filtered.q05.H2(dt)    = quantile(H2,0.025);
                filtered.q05.H3(dt)    = quantile(H3,0.025);

                filtered.q95.H1(dt)    = quantile(H1,0.975);
                filtered.q95.H2(dt)    = quantile(H2,0.975);
                filtered.q95.H3(dt)    = quantile(H3,0.975);
            end % end dt = 1:T
        end % end logLikelihood


        %% Function updateState: updates the state
        %  Input:
        %    - dt:                  Day index
        %    - pv:                  Parameter value structure
        %    - H1old:               Previous values of first factor
        %    - H2old:               Previous values of second factor
        %    - H3old:               Previous values of third factor
        %  Output:
        %    - weights:             Weights
        %    - weights_likelihood:  Weights for likelihood calculation
        %    - H1:                  First factor (post update)
        %    - H2:                  Second factor (post update)
        %    - H3:                  Third factor (post update)
        %    - XP:                  Positive jumps
        %    - XN:                  Negative jumps
        %    - Y:                   Independent jumps
        %
        function [weights, weights_loglikelihood, H1, H2, H3, XP, XN, Y] = updateState(self, dt, pv, H1old, H2old, H3old)
            delta_t   = 1/252;

            [HPold, HNold]   = self.model.getJumpIntensity(pv,H1old,H2old,H3old);
            HPold = HPold*delta_t;
            HNold = HNold*delta_t;
            HYold = HNold;

            % generate the particles
            idt                       = mod(dt-1,1000)+1;
            innovations.psi1          = self.particles.psi1(:,idt).*sqrt(H1old*delta_t);
            innovations.psi2          = self.particles.psi2(:,idt).*sqrt(H2old*delta_t);
            innovations.psi3          = self.particles.psi3(:,idt).*sqrt(H3old*delta_t);

            % use importance sampling (factor of 10) to artificially
            % increase the number of jumps
            innovations.Np          = (self.particles.np(:,idt) < min(0.99,HPold.*10)).*1;
            innovations.Nn          = (self.particles.nn(:,idt) < min(0.99,HNold.*10)).*1;
            innovations.Ny          = (self.particles.ny(:,idt) < min(0.99,HYold.*10)).*1;
            innovations.xp          =  innovations.Np.*expinv(self.particles.xp(:,idt),1./pv.lambdap);
            innovations.xn          = -innovations.Nn.*expinv(self.particles.xn(:,idt),1./pv.lambdan);
            innovations.y           = -innovations.Ny.*expinv(self.particles.xy(:,idt),1./pv.lambdan);

            % compute convexity correction for jumps
            xip                     =  1/ (pv.lambdap - 1);
            xin                     = -1/ (pv.lambdan + 1);

            XP                      = innovations.xp;
            XN                      = innovations.xn;
            Y                       = innovations.y;

            H1      = boxed(H1old + pv.kappa1*(pv.nu1 - H1old)*delta_t + pv.sigma1*(innovations.psi1 ) + pv.mu1* XN.^2, self.model.H_MIN, self.model.H_MAX);
            H2      = boxed(H2old + pv.kappa2*(pv.nu2 - H2old)*delta_t + pv.sigma2*(innovations.psi2 ) , self.model.H_MIN, self.model.H_MAX);
            H3      = boxed(H3old + pv.kappa3.* (-H3old)*delta_t + pv.mu3*((1-pv.rho3).*XN.^2+ pv.rho3.*(Y.^2)), self.model.H_MIN, self.model.H_MAX);

            % compute risk-neutral moments (and factors needed to compute the
            % risk-neutral variances)
            [Volatility,Skewness,Kurtosis]     = self.model.getMoments(H1,H2,H3,false);
            Variance = Volatility.^2;

            % calculate the noisy cumulants
            cum2    = Variance;
            cum3    = Skewness.*Variance.^1.5;
            cum4    = (Kurtosis-3).*Variance.^2;

            % compute the weights
            lweights   = 0;
            if ~isempty(self.model.moments.Variance(dt,:))
                mm       = repmat(self.model.moments.Variance(dt,:), self.nb_particles,1);
                ss       = repmat(pv.del2.*std(self.model.moments.Variance), self.nb_particles,1);
                lweights = lweights + sum(log(normpdf(cum2, mm, ss)),2);
            end
            if ~isempty(self.model.moments.Skewness(dt,:))
                mm       = repmat(self.model.moments.Skewness(dt,:).*self.model.moments.Variance(dt,:).^1.5, self.nb_particles,1);
                ss       = repmat(pv.del3.*std(self.model.moments.Skewness.*self.model.moments.Variance.^1.5), self.nb_particles,1);
                lweights = lweights + sum(log(normpdf(cum3, mm, ss)),2);
            end
            if ~isempty(self.model.moments.Kurtosis(dt,:))
                mm       = repmat((self.model.moments.Kurtosis(dt,:)-3).*self.model.moments.Variance(dt,:).^2, self.nb_particles,1);
                ss       = repmat(pv.del4.*std((self.model.moments.Kurtosis-3).*self.model.moments.Variance.^2), self.nb_particles,1);
                lweights = lweights + sum(log(normpdf(cum4, mm, ss)),2);
            end

            % correct the log-likelihood for importance sampling
            lweights   = lweights + innovations.Np.*(log(HPold) - log(min(0.99,10.*HPold))) + ...
                (1-innovations.Np).*(log(1-HPold) - log(1-min(0.99,10.*HPold)));
            lweights   = lweights + innovations.Nn.*(log(HNold) - log(min(0.99,10.*HNold))) + ...
                (1-innovations.Nn).*(log(1-HNold) - log(1-min(0.99,10.*HNold)));
            lweights   = lweights + innovations.Ny.*(log(HYold) - log(min(0.99,10.*HYold))) + ...
                (1-innovations.Ny).*(log(1-HYold) - log(1-min(0.99,10.*HYold)));

            weights_loglikelihood = lweights;

            weights_loglikelihood(isnan(weights_loglikelihood)) = -Inf;
            lweights(isnan(lweights)) = -Inf;
            weights                   = exp(lweights-max(lweights));
        end % end updateState

    end % end methods

end % end classdef ParticleFilter
