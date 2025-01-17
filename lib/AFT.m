%%  Class AFT
%   This class contains functions and utility to create the estimation method
%   proposed in Bégin and Zafari (2025).
classdef AFT < GARCH
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           PROPERTIES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties
        moments     =  [];         % Structure to store risk-neutral moments
        coeffs      =  [];         % Structure to store coefficients needed to compute model-based moments
    end % end properties

    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %           METHODS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Access = private) % private methods
        function  [alpha_2,beta_2,alpha_3,beta_3,B4,Alpha4] = AFT_RNM_Coeff(self, tau)
            % copy of parameters (in a structure)
            pv    = self.getPV();

            eta   = pv.eta;        c0_m = pv.cn0;        c3_p = pv.cp3;
            rho_1 = pv.rho1;       vb_1 = pv.nu1;     kappa_1 = pv.kappa1;  sigma_1 = pv.sigma1;
            rho_2 = pv.rho2;       vb_2 = pv.nu2;     kappa_2 = pv.kappa2;  sigma_2 = pv.sigma2;
            mu_3  = pv.mu3;     kappa_3 = pv.kappa3;    rho_3 = pv.rho3;
            c0_p  = pv.cp0;        c1_m = pv.cn1;        c1_p = pv.cp1;        c2_m = pv.cn2;    c2_p = pv.cp2;   c3_m = pv.cn3;
            lamd_m= pv.lambdan;  lamd_p = pv.lambdap;    mu_1 = pv.mu1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % following the Appendix Feunou and Okou 2016
            Theta_nc   =   self.Theta_nc_Derivative(mu_1,mu_3,rho_3,lamd_m);
            Theta_ni   =   self.Theta_ni_Derivative(mu_3,rho_3,lamd_m);
            Theta_p    =   self.Theta_p_Derivative(lamd_p);

            %%%
            B          =   [-0.5 + c1_m * Theta_nc.q0 + c1_p * Theta_p.q0 - c1_m * (lamd_m/(1 + lamd_m)-1) - c1_p * (lamd_p/(lamd_p-1)-1);...
                -0.5 * (eta^2) + c3_m * Theta_nc.q0 + c3_p * Theta_p.q0 - c3_m * (lamd_m/(1 + lamd_m)-1) - c3_p * (lamd_p/(lamd_p-1)-1);...
                -0.5 + c2_m * Theta_nc.q0 + c2_p * Theta_p.q0 - c2_m * (lamd_m/(1 + lamd_m)-1) - c2_p * (lamd_p/(lamd_p-1)-1)];
            A          =   [-kappa_1 + c1_m * Theta_nc.q1, c1_m * Theta_nc.q3 + c1_m * Theta_ni.q3, 0; c3_m * Theta_nc.q1, -kappa_3 + c3_m * Theta_nc.q3 + c3_m * Theta_ni.q3, 0; c2_m * Theta_nc.q1, c2_m * Theta_nc.q3 + c2_m * Theta_ni.q3, -kappa_2];
            Aalpha     =   [kappa_1 * vb_1 + c0_m * Theta_nc.q1; c0_m * Theta_nc.q3 + c0_m * Theta_ni.q3; kappa_2 * vb_2];

            %%%
            [VA,DA]    =   eig(A);
            P          =   [VA(:,2), VA(:,3), VA(:,1)];
            D          =   diag([DA(2,2),DA(3,3),DA(1,1)]);
            lamd_1     =   D(1,1);
            lamd_2     =   D(2,2);
            lamd_3     =   D(3,3);
            P12        =   P(1:2,1:2);
            P3         =   P(3,1:2).';
            P          =   [P12, zeros(2,1);P3.', 1];
            P12_i      =   (1/(P12(1,1) * P12(2,2) - P12(2,1) * P12(1,2))) * [P12(2,2),-P12(1,2);-P12(2,1),P12(1,1)];

            % inverse of P and D
            P_i        =   [P12_i, zeros(2,1); -P3.' * P12_i, 1];
            D_i        =   [1/lamd_1 0 0; 0 1/lamd_2 0; 0 0 1/lamd_3];

            %%%%%%%%%%%%%%%%%%%%  beta_1 %%%%%%%%%%%%%%%%%%%%%%%
            A_i        =   P * D_i * P_i;
            C0         =   A_i * B;
            beta_1     =   expm(tau * A) * C0 - C0;
            Cb0        =   P_i * C0;
            %%%%%%%%%%%%%%%%%%%%  alpha_2, beta_2 %%%%%%%%%%%%%%%%%%%%%%%
            B2_1c      =  1 + c1_m * Theta_nc.q0q0 + c1_p * Theta_p.q0q0;
            B2_3c      =  eta^2 + c3_m * Theta_nc.q0q0 + c3_p * Theta_p.q0q0;
            B2_2c      =  1 + c2_m * Theta_nc.q0q0 + c2_p * Theta_p.q0q0;

            %%%
            B2_1s      =  [2 * sigma_1 * rho_1 + 2 * c1_m * Theta_nc.q0q1; 2 * c1_m * Theta_nc.q0q3; 0];
            B2_3s      =  [2 * c3_m * Theta_nc.q0q1; 2 * c3_m * Theta_nc.q0q3; 0];
            B2_2s      =  [2 * c2_m * Theta_nc.q0q1; 2 * c2_m * Theta_nc.q0q3; 2 * sigma_2 * rho_2];

            %%%
            B2_1q      =  [sigma_1^2 + c1_m * Theta_nc.q1q1, c1_m * Theta_nc.q1q3, 0; c1_m * Theta_nc.q1q3, c1_m * Theta_nc.q3q3 + c1_m * Theta_ni.q3q3, 0; 0, 0, 0];
            B2_3q      =  [c3_m * Theta_nc.q1q1, c3_m * Theta_nc.q1q3, 0; c3_m * Theta_nc.q1q3, c3_m * Theta_nc.q3q3 + c3_m * Theta_ni.q3q3, 0; 0, 0, 0];
            B2_2q      =  [c2_m * Theta_nc.q1q1, c2_m * Theta_nc.q1q3, 0; c2_m * Theta_nc.q1q3, c2_m * Theta_nc.q3q3 + c2_m * Theta_ni.q3q3, 0; 0, 0, sigma_2^2];

            %%%
            phi_1c     =  B2_1c - B2_1s.' * C0 + C0.' * B2_1q * C0;     phi_3c   =  B2_3c - B2_3s.' * C0 + C0.' * B2_3q * C0;        phi_2c = B2_2c - B2_2s.' * C0 + C0.' * B2_2q * C0;       phi_c = [phi_1c; phi_3c; phi_2c];   phib_c = A_i*phi_c;
            phi_1s     =  B2_1s - 2 * B2_1q * C0;                     phi_3s   =  B2_3s - 2 * B2_3q * C0;                        phi_2s = B2_2s - 2 * B2_2q * C0;                       phi_s = P_i * [phi_1s.';phi_3s.';phi_2s.']*P;

            %%%% Part 1
            beta_2_1   =  expm(tau*A) * phib_c - phib_c;
            %%%% Part 2
            beta_2_2      = P * self.intcubmat(tau,D,0,phi_s) * Cb0;
            %%%% Part 3
            Bb2_1q     =  P.' * B2_1q * P;        Bb2_3q  = P.' * B2_3q*P;    Bb2_2q  =  P.' * B2_2q * P;
            Bt2_1q     =  P_i(1,1) * Bb2_1q + P_i(1,2) * Bb2_3q;
            Bt2_3q     =  P_i(2,1) * Bb2_1q + P_i(2,2) * Bb2_3q;
            Bt2_2q     =  P_i(3,1) * Bb2_1q + P_i(3,2) * Bb2_3q + Bb2_2q;
            beta_2_3   =  P * self.SecondMom3(tau,Bt2_1q,Bt2_3q,Bt2_2q,D,Cb0);
            %%%% the sum
            beta_2     =  beta_2_1 + beta_2_2 + beta_2_3;

            %%%% compute int_beta_2
            Abalpha    =  A_i.' * Aalpha;
            int_beta_1 =  A_i * beta_1 - tau * C0;
            int_beta_2 =  A_i * (beta_2 - tau * [B2_1c;B2_3c;B2_2c] - ...
                [B2_1s.';B2_3s.';B2_2s.'] * int_beta_1 + ...
                2 * [C0.' * B2_1q; C0.' * B2_3q; C0.' * B2_2q] * A_i * beta_1 - tau * [C0.' * B2_1q * C0; C0.' * B2_3q * C0; C0.' * B2_2q * C0] - [C0.' * self.intsquaremat(P,P_i,D,tau,B2_1q) * C0; C0.' * self.intsquaremat(P,P_i,D,tau,B2_3q) * C0; C0.' * self.intsquaremat(P,P_i,D,tau,B2_2q) * C0]);

            %%%% alpha_2
            alpha2_c   =  c0_m * Theta_nc.q0q0 + c0_p * Theta_p.q0q0;
            alpha2_s   =  [2 * c0_m * Theta_nc.q0q1; 2 * c0_m * Theta_nc.q0q3; 0];
            alpha2_q   =  [c0_m * Theta_nc.q1q1, c0_m * Theta_nc.q1q3, 0; c0_m * Theta_nc.q1q3, c0_m * Theta_nc.q3q3 + c0_m * Theta_ni.q3q3, 0; 0, 0, 0];
            phi_0c     =  alpha2_c - alpha2_s.' * C0 + C0.' * alpha2_q * C0;            phi_0s  =  alpha2_s - 2 * alpha2_q * C0;
            psi0       =  phi_0c - Abalpha.' * phi_c;                                    psi1  =  phi_0s - [phi_1s.'; phi_3s.'; phi_2s.'].' * Abalpha;
            alpha_2    =  tau * psi0 + psi1.' * A_i * beta_1 + C0.' * self.intsquaremat(P,P_i,D,tau,alpha2_q) * C0 + Abalpha.' * (beta_2 - [C0.' * self.intsquaremat(P,P_i,D,tau,B2_1q) * C0; C0.' * self.intsquaremat(P,P_i,D,tau,B2_3q) * C0; C0.' * self.intsquaremat(P,P_i,D,tau,B2_2q) * C0]);


            %%%%%%%%%%%%%%%%%%%%  alpha_3, beta_3 %%%%%%%%%%%%%%%%%%%%%%%
            B3_1c      =  c1_m * Theta_nc.q0q0q0 + c1_p * Theta_p.q0q0q0;
            B3_3c      =  c3_m * Theta_nc.q0q0q0 + c3_p * Theta_p.q0q0q0;
            B3_2c      =  c2_m * Theta_nc.q0q0q0 + c2_p * Theta_p.q0q0q0;
            alpha3_c   =  c0_m * Theta_nc.q0q0q0 + c0_p * Theta_p.q0q0q0;
            alphab3_c  =  alpha3_c - Abalpha(1) * B3_1c - Abalpha(2) * B3_3c - Abalpha(3) * B3_2c;

            %%%
            B3_1s      =  [3 * c1_m * Theta_nc.q0q0q1; 3 * c1_m * Theta_nc.q0q0q3; 0];
            B3_3s      =  [3 * c3_m * Theta_nc.q0q0q1; 3 * c3_m * Theta_nc.q0q0q3; 0];
            B3_2s      =  [3 * c2_m * Theta_nc.q0q0q1; 3 * c2_m * Theta_nc.q0q0q3; 0];
            alpha3_s   =  [3 * c0_m * Theta_nc.q0q0q1; 3 * c0_m * Theta_nc.q0q0q3; 0];
            alphab3_s  =  alpha3_s - Abalpha(1) * B3_1s - Abalpha(2) * B3_3s - Abalpha(3) * B3_2s;

            %%%
            B3_1qc     =  [3 * c1_m * Theta_nc.q0q1q1, 3 * c1_m * Theta_nc.q0q1q3, 0; 3 * c1_m * Theta_nc.q0q1q3, 3 * c1_m * Theta_nc.q0q3q3, 0; 0, 0, 0];
            B3_3qc     =  [3 * c3_m * Theta_nc.q0q1q1, 3 * c3_m * Theta_nc.q0q1q3, 0; 3 * c3_m * Theta_nc.q0q1q3, 3 * c3_m * Theta_nc.q0q3q3, 0; 0, 0, 0];
            B3_2qc     =  [3 * c2_m * Theta_nc.q0q1q1, 3 * c2_m * Theta_nc.q0q1q3, 0; 3 * c2_m * Theta_nc.q0q1q3, 3 * c2_m * Theta_nc.q0q3q3, 0; 0, 0, 0];
            alpha3_qc  =  [3 * c0_m * Theta_nc.q0q1q1, 3 * c0_m * Theta_nc.q0q1q3, 0; 3 * c0_m * Theta_nc.q0q1q3, 3 * c0_m * Theta_nc.q0q3q3, 0; 0, 0, 0];
            alphab3_qc =  alpha3_qc - Abalpha(1) * B3_1qc - Abalpha(2) * B3_3qc - Abalpha(3) * B3_2qc;

            %%%
            Bb3_1qc    =  P.' * B3_1qc * P;     Bb3_3qc  =  P.' * B3_3qc * P;            Bb3_2qc  =  P.' * B3_2qc*P;      Bb3_qc  =  P_i * [Cb0.' * Bb3_1qc; Cb0.' * Bb3_3qc; Cb0.' * Bb3_2qc];
            Bt3_1qc    =  P_i(1,1) * Bb3_1qc + P_i(1,2) * Bb3_3qc;
            Bt3_3qc    =  P_i(2,1) * Bb3_1qc + P_i(2,2) * Bb3_3qc;
            Bt3_2qc    =  P_i(3,1) * Bb3_1qc + P_i(3,2) * Bb3_3qc + Bb3_2qc;

            %%%
            B3_1_1qs   =  [c1_m * Theta_nc.q1q1q1; 3 * c1_m * Theta_nc.q1q1q3; 0];    Bb3_1_1qs  =  P.'*B3_1_1qs;
            B3_3_1qs   =  [3 * c1_m * Theta_nc.q1q3q3; c1_m * Theta_nc.q3q3q3 + c1_m * Theta_ni.q3q3q3;0];            Bb3_3_1qs = P.' * B3_3_1qs;
            B3_2_1qs   =  [0;0;0];     Bb3_2_1qs = P.' * B3_2_1qs;

            %%%
            alpha3_1_qs  =  [c0_m * Theta_nc.q1q1q1; 3 * c0_m * Theta_nc.q1q1q3; 0];
            alpha3_3_qs  =  [3 * c0_m * Theta_nc.q1q3q3; c0_m * Theta_nc.q3q3q3 + c0_m * Theta_ni.q3q3q3; 0];
            alpha3_2_qs  =  [0;0;0];

            %%%
            B3_1_3qs     =  [c3_m * Theta_nc.q1q1q1; 3 * c3_m * Theta_nc.q1q1q3; 0];                                Bb3_1_3qs = P.' * B3_1_3qs;
            B3_3_3qs     =  [3 * c3_m * Theta_nc.q1q3q3; c3_m * Theta_nc.q3q3q3 + c3_m * Theta_ni.q3q3q3; 0];       Bb3_3_3qs = P.' * B3_3_3qs;
            B3_2_3qs     =  [0;0;0];           Bb3_2_3qs = P.' * B3_2_3qs;

            %%%
            B3_1_2qs     =  [c2_m * Theta_nc.q1q1q1; 3 * c2_m * Theta_nc.q1q1q3; 0];                                Bb3_1_2qs = P.' * B3_1_2qs;
            B3_3_2qs     =  [3 * c2_m * Theta_nc.q1q3q3; c2_m * Theta_nc.q3q3q3 + c2_m * Theta_ni.q3q3q3;0];        Bb3_3_2qs = P.' * B3_3_2qs;
            B3_2_2qs     =  [0;0;0];                                                                                Bb3_2_2qs = P.' * B3_2_2qs;

            %%%
            alphab3_1_qs  =  alpha3_1_qs - Abalpha(1) * B3_1_1qs - Abalpha(2) * B3_1_3qs - Abalpha(3) * B3_1_2qs;
            alphab3_3_qs  =  alpha3_3_qs - Abalpha(1) * B3_3_1qs - Abalpha(2) * B3_3_3qs - Abalpha(3) * B3_3_2qs;
            alphab3_2_qs  =  alpha3_2_qs - Abalpha(1) * B3_2_1qs - Abalpha(2) * B3_2_3qs - Abalpha(3) * B3_2_2qs;

            %%%
            B3_11_1qs     = P(1,1) * P(1,1) * Bb3_1_1qs + P(2,1) * P(2,1) * Bb3_3_1qs + P(3,1) * P(3,1) * Bb3_2_1qs;
            B3_12_1qs     = P(1,1) * P(1,2) * Bb3_1_1qs + P(2,1) * P(2,2) * Bb3_3_1qs + P(3,1) * P(3,2) * Bb3_2_1qs;
            B3_13_1qs     = P(1,1) * P(1,3) * Bb3_1_1qs + P(2,1) * P(2,3) * Bb3_3_1qs + P(3,1) * P(3,3) * Bb3_2_1qs;
            B3_21_1qs     = P(1,2) * P(1,1) * Bb3_1_1qs + P(2,2) * P(2,1) * Bb3_3_1qs + P(3,2) * P(3,1) * Bb3_2_1qs;
            B3_22_1qs     = P(1,2) * P(1,2) * Bb3_1_1qs + P(2,2) * P(2,2) * Bb3_3_1qs + P(3,2) * P(3,2) * Bb3_2_1qs;
            B3_23_1qs     = P(1,2) * P(1,3) * Bb3_1_1qs + P(2,2) * P(2,3) * Bb3_3_1qs + P(3,2) * P(3,3) * Bb3_2_1qs;
            B3_31_1qs     = P(1,3) * P(1,1) * Bb3_1_1qs + P(2,3) * P(2,1) * Bb3_3_1qs + P(3,3) * P(3,1) * Bb3_2_1qs;
            B3_32_1qs     = P(1,3) * P(1,2) * Bb3_1_1qs + P(2,3) * P(2,2) * Bb3_3_1qs + P(3,3) * P(3,2) * Bb3_2_1qs;
            B3_33_1qs     = P(1,3) * P(1,3) * Bb3_1_1qs + P(2,3) * P(2,3) * Bb3_3_1qs + P(3,3) * P(3,3) * Bb3_2_1qs;

            %%%
            B3_11_3qs     = P(1,1) * P(1,1) * Bb3_1_3qs + P(2,1) * P(2,1) * Bb3_3_3qs + P(3,1) * P(3,1) * Bb3_2_3qs;
            B3_12_3qs     = P(1,1) * P(1,2) * Bb3_1_3qs + P(2,1) * P(2,2) * Bb3_3_3qs + P(3,1) * P(3,2) * Bb3_2_3qs;
            B3_13_3qs     = P(1,1) * P(1,3) * Bb3_1_3qs + P(2,1) * P(2,3) * Bb3_3_3qs + P(3,1) * P(3,3) * Bb3_2_3qs;
            B3_21_3qs     = P(1,2) * P(1,1) * Bb3_1_3qs + P(2,2) * P(2,1) * Bb3_3_3qs + P(3,2) * P(3,1) * Bb3_2_3qs;
            B3_22_3qs     = P(1,2) * P(1,2) * Bb3_1_3qs + P(2,2) * P(2,2) * Bb3_3_3qs + P(3,2) * P(3,2) * Bb3_2_3qs;
            B3_23_3qs     = P(1,2) * P(1,3) * Bb3_1_3qs + P(2,2) * P(2,3) * Bb3_3_3qs + P(3,2) * P(3,3) * Bb3_2_3qs;
            B3_31_3qs     = P(1,3) * P(1,1) * Bb3_1_3qs + P(2,3) * P(2,1) * Bb3_3_3qs + P(3,3) * P(3,1) * Bb3_2_3qs;
            B3_32_3qs     = P(1,3) * P(1,2) * Bb3_1_3qs + P(2,3) * P(2,2) * Bb3_3_3qs + P(3,3) * P(3,2) * Bb3_2_3qs;
            B3_33_3qs     = P(1,3) * P(1,3) * Bb3_1_3qs + P(2,3) * P(2,3) * Bb3_3_3qs + P(3,3) * P(3,3) * Bb3_2_3qs;

            %%%
            B3_11_2qs     = P(1,1) * P(1,1) * Bb3_1_2qs + P(2,1) * P(2,1) * Bb3_3_2qs + P(3,1) * P(3,1) * Bb3_2_2qs;
            B3_12_2qs     = P(1,1) * P(1,2) * Bb3_1_2qs + P(2,1) * P(2,2) * Bb3_3_2qs + P(3,1) * P(3,2) * Bb3_2_2qs;
            B3_13_2qs     = P(1,1) * P(1,3) * Bb3_1_2qs + P(2,1) * P(2,3) * Bb3_3_2qs + P(3,1) * P(3,3) * Bb3_2_2qs;
            B3_21_2qs     = P(1,2) * P(1,1) * Bb3_1_2qs + P(2,2) * P(2,1) * Bb3_3_2qs + P(3,2) * P(3,1) * Bb3_2_2qs;
            B3_22_2qs     = P(1,2) * P(1,2) * Bb3_1_2qs + P(2,2) * P(2,2) * Bb3_3_2qs + P(3,2) * P(3,2) * Bb3_2_2qs;
            B3_23_2qs     = P(1,2) * P(1,3) * Bb3_1_2qs + P(2,2) * P(2,3) * Bb3_3_2qs + P(3,2) * P(3,3) * Bb3_2_2qs;
            B3_31_2qs     = P(1,3) * P(1,1) * Bb3_1_2qs + P(2,3) * P(2,1) * Bb3_3_2qs + P(3,3) * P(3,1) * Bb3_2_2qs;
            B3_32_2qs     = P(1,3) * P(1,2) * Bb3_1_2qs + P(2,3) * P(2,2) * Bb3_3_2qs + P(3,3) * P(3,2) * Bb3_2_2qs;
            B3_33_2qs     = P(1,3) * P(1,3) * Bb3_1_2qs + P(2,3) * P(2,3) * Bb3_3_2qs + P(3,3) * P(3,3) * Bb3_2_2qs;

            %%%
            B3_c_1qs      =  diag([B3_1_1qs.' * C0; B3_3_1qs.' * C0; B3_2_1qs.' * C0]);      Bb3_c_1qs = diag([Bb3_1_1qs.' * C0; Bb3_3_1qs.' * C0; Bb3_2_1qs.' * C0]);
            B3_c_3qs      =  diag([B3_1_3qs.' * C0; B3_3_3qs.' * C0; B3_2_3qs.' * C0]);      Bb3_c_3qs = diag([Bb3_1_3qs.' * C0; Bb3_3_3qs.' * C0; Bb3_2_3qs.' * C0]);
            B3_c_2qs      =  diag([B3_1_2qs.' * C0; B3_3_2qs.' * C0; B3_2_2qs.' * C0]);      Bb3_c_2qs = diag([Bb3_1_2qs.' * C0; Bb3_3_2qs.' * C0; Bb3_2_2qs.' * C0]);

            Bt3_c_1qs     =  P_i(1,1) * Bb3_c_1qs + P_i(1,2) * Bb3_c_3qs;
            Bt3_c_3qs     =  P_i(2,1) * Bb3_c_1qs + P_i(2,2) * Bb3_c_3qs;
            Bt3_c_2qs     =  P_i(3,1) * Bb3_c_1qs + P_i(3,2) * Bb3_c_3qs + Bb3_c_2qs;

            Bh3_1_1qs     = Cb0(1) * (Cb0(1) * B3_11_1qs + Cb0(2) * B3_21_1qs + Cb0(3) * B3_31_1qs);
            Bh3_2_1qs     = Cb0(2) * (Cb0(1) * B3_12_1qs + Cb0(2) * B3_22_1qs + Cb0(3) * B3_32_1qs);
            Bh3_3_1qs     = Cb0(3) * (Cb0(1) * B3_13_1qs + Cb0(2) * B3_23_1qs + Cb0(3) * B3_33_1qs);

            Bh3_1_3qs     = Cb0(1) * (Cb0(1) * B3_11_3qs + Cb0(2) * B3_21_3qs + Cb0(3) * B3_31_3qs);
            Bh3_2_3qs     = Cb0(2) * (Cb0(1) * B3_12_3qs + Cb0(2) * B3_22_3qs + Cb0(3) * B3_32_3qs);
            Bh3_3_3qs     = Cb0(3) * (Cb0(1) * B3_13_3qs + Cb0(2) * B3_23_3qs + Cb0(3) * B3_33_3qs);

            Bh3_1_2qs     = Cb0(1) * (Cb0(1) * B3_11_2qs + Cb0(2) * B3_21_2qs + Cb0(3) * B3_31_2qs);
            Bh3_2_2qs     = Cb0(2) * (Cb0(1) * B3_12_2qs + Cb0(2) * B3_22_2qs + Cb0(3) * B3_32_2qs);
            Bh3_3_2qs     = Cb0(3) * (Cb0(1) * B3_13_2qs + Cb0(2) * B3_23_2qs + Cb0(3) * B3_33_2qs);

            Gam2_0        = P_i * [(Bh3_1_1qs + Bh3_2_1qs + Bh3_3_1qs + 2 * Bb3_c_1qs.' * Cb0).'; (Bh3_1_3qs + Bh3_2_3qs + Bh3_3_3qs + 2 * Bb3_c_3qs.' * Cb0).'; (Bh3_1_2qs + Bh3_2_2qs + Bh3_3_2qs + 2 * Bb3_c_2qs.' * Cb0).'];
            Gam2_1        = P_i * [Bh3_1_1qs.'; Bh3_1_3qs.'; Bh3_1_2qs.'];
            Gam2_2        = P_i * [Bh3_2_1qs.'; Bh3_2_3qs.'; Bh3_2_2qs.'];
            Gam2_3        = P_i * [Bh3_3_1qs.'; Bh3_3_3qs.'; Bh3_3_2qs.'];

            Gam2_11       = P_i * [B3_11_1qs.'; B3_11_3qs.'; B3_11_2qs.'];
            Gam2_12       = P_i * [B3_12_1qs.'; B3_12_3qs.'; B3_12_2qs.'];
            Gam2_13       = P_i * [B3_13_1qs.'; B3_13_3qs.'; B3_13_2qs.'];
            Gam2_21       = P_i * [B3_21_1qs.'; B3_21_3qs.'; B3_21_2qs.'];
            Gam2_22       = P_i * [B3_22_1qs.'; B3_22_3qs.'; B3_22_2qs.'];
            Gam2_23       = P_i * [B3_23_1qs.'; B3_23_3qs.'; B3_23_2qs.'];
            Gam2_31       = P_i * [B3_31_1qs.'; B3_31_3qs.'; B3_31_2qs.'];
            Gam2_32       = P_i * [B3_32_1qs.'; B3_32_3qs.'; B3_32_2qs.'];
            Gam2_33       = P_i * [B3_33_1qs.'; B3_33_3qs.'; B3_33_2qs.'];


            %%%
            B3_1hc        =  [3 * sigma_1 * rho_1 + 3 * c1_m * Theta_nc.q0q1; 3 * c1_m * Theta_nc.q0q3;0];
            B3_3hc        =  [3 * c3_m * Theta_nc.q0q1; 3 * c3_m * Theta_nc.q0q3; 0];
            B3_2hc        =  [3 * c2_m * Theta_nc.q0q1; 3 * c2_m * Theta_nc.q0q3; 3 * sigma_2 * rho_2];
            alpha3_hc     =  [3 * c0_m * Theta_nc.q0q1; 3 * c0_m * Theta_nc.q0q3; 0];
            alphab3_hc    =  alpha3_hc - Abalpha(1) * B3_1hc - Abalpha(2) * B3_3hc - Abalpha(3) * B3_2hc;

            %%%
            B3_1hs        = [3 * sigma_1^2 + 3 * c1_m * Theta_nc.q1q1, 3 * c1_m * Theta_nc.q1q3, 0; 3 * c1_m * Theta_nc.q1q3, 3 * c1_m * Theta_nc.q3q3 + 3 * c1_m * Theta_ni.q3q3, 0; 0, 0, 0];
            B3_3hs        = [3 * c3_m * Theta_nc.q1q1, 3 * c3_m * Theta_nc.q1q3, 0; 3 * c3_m * Theta_nc.q1q3, 3 * c3_m * Theta_nc.q3q3 + 3 * c3_m * Theta_ni.q3q3, 0; 0, 0, 0];
            B3_2hs        = [3 * c2_m * Theta_nc.q1q1, 3 * c2_m * Theta_nc.q1q3, 0; 3 * c2_m * Theta_nc.q1q3, 3 * c2_m * Theta_nc.q3q3 + 3 * c2_m * Theta_ni.q3q3, 0; 0, 0, 3 * sigma_2^2];
            alpha3_hs     = [3 * c0_m * Theta_nc.q1q1, 3 * c0_m * Theta_nc.q1q3, 0; 3 * c0_m * Theta_nc.q1q3, 3 * c0_m * Theta_nc.q3q3 + 3 * c0_m * Theta_ni.q3q3, 0; 0, 0, 0];
            alphab3_hs    = alpha3_hs - Abalpha(1) * B3_1hs - Abalpha(2) * B3_3hs - Abalpha(3) * B3_2hs;
            Bb3_1hs       = P.' * B3_1hs * P;      Bb3_3hs = P.' * B3_3hs * P;       Bb3_2hs = P.' * B3_2hs * P;    E_1 = [1,0,0;0,0,0;0,0,0];     E_2 = [0,0,0; 0,1,0; 0,0,0];     E_3 = [0,0,0; 0,0,0; 0,0,1];

            B3_0_hc       = P_i * [(B3_1hc - B3_1hs * C0).'; (B3_3hc - B3_3hs * C0).'; (B3_2hc - B3_2hs * C0).'] * P;
            B3_1_hc       = P_i * [Cb0.' * E_1 * Bb3_1hs; Cb0.' * E_1 * Bb3_3hs; Cb0.' * E_1 * Bb3_2hs];
            B3_2_hc       = P_i * [Cb0.' * E_2 * Bb3_1hs; Cb0.' * E_2 * Bb3_3hs; Cb0.' * E_2 * Bb3_2hs];
            B3_3_hc       = P_i * [Cb0.' * E_3 * Bb3_1hs; Cb0.' * E_3 * Bb3_3hs; Cb0.' * E_3 * Bb3_2hs];

            B3_1q         = B3_1qc + [B3_1_1qs.' * beta_1, 0,0;0,B3_3_1qs.' * beta_1,0; 0,0,0];
            B3_3q         = B3_3qc + [B3_1_3qs.' * beta_1, 0,0;0,B3_3_3qs.' * beta_1,0; 0,0,0];
            B3_2q         = B3_2qc + [B3_1_2qs.' * beta_1, 0,0;0,B3_3_2qs.' * beta_1,0; 0,0,0];
            alpha3_q      = alpha3_qc + [alpha3_1_qs.' * beta_1, 0,0; 0,alpha3_3_qs.' * beta_1,0; 0,0,0];

            B3_1          = B3_1c + B3_1s.' * beta_1 + beta_1.' * B3_1q * beta_1 + (B3_1hc + B3_1hs * beta_1).' * beta_2;
            B3_3          = B3_3c + B3_3s.' * beta_1 + beta_1.' * B3_3q * beta_1 + (B3_3hc + B3_3hs * beta_1).' * beta_2;
            B3_2          = B3_2c + B3_2s.' * beta_1 + beta_1.' * B3_2q * beta_1 + (B3_2hc + B3_2hs * beta_1).' * beta_2;
            Alpha3        = alpha3_c + alpha3_s.' * beta_1 + beta_1.' * alpha3_q * beta_1 + (alpha3_hc + alpha3_hs * beta_1).' * beta_2;
            B3            = [B3_1; B3_3; B3_2];

            %%%% Part 1
            B3_c          = [B3_1c; B3_3c; B3_2c];                     B3_s  = [B3_1s.'; B3_3s.'; B3_2s.'];       Bb3_s = P_i * B3_s * P;        Bb3_c = A_i * (B3_c - B3_s * C0);
            beta_3_1      = P * self.intcubmat(tau,D,0,Bb3_s) * Cb0;
            beta_3_1      = beta_3_1 + expm(tau * A) * Bb3_c - Bb3_c;
            %%%% Part 2
            %%%% Part 2-1
            beta_3_2_1    = A_i * ([Cb0.' * Bb3_1qc; Cb0.' * Bb3_3qc; Cb0.' * Bb3_2qc] * Cb0);
            beta_3_2_1    = expm(tau * A) * beta_3_2_1 - beta_3_2_1;
            beta_3_2_1    = beta_3_2_1 - 2 * P * self.intcubmat(tau,D,0,Bb3_qc) * Cb0;
            beta_3_2_1_2  = P * self.SecondMom3(tau,Bt3_1qc,Bt3_3qc,Bt3_2qc,D,Cb0);
            beta_3_2_1    = beta_3_2_1 + beta_3_2_1_2;
            %%%% Part 2-2
            beta_3_2_2    = A_i * [C0.' * B3_c_1qs * C0; C0.' * B3_c_3qs * C0;C0.' * B3_c_2qs * C0];
            beta_3_2_2    = beta_3_2_2 - expm(tau * A) * beta_3_2_2;
            beta_3_2_2    = beta_3_2_2 + P * self.intcubmat(tau,D,0,Gam2_0) * Cb0 - 2 * P * self.intcubmat(tau,D,D(1,1),Gam2_1) * Cb0 - 2 * P * self.intcubmat(tau,D,D(2,2),Gam2_2) * Cb0 - 2 * P * self.intcubmat(tau,D,D(3,3),Gam2_3) * Cb0;
            beta_3_2_2_2  = P * self.SecondMom3(tau,Bt3_c_1qs,Bt3_c_3qs,Bt3_c_2qs,D,Cb0);
            beta_3_2_2    = beta_3_2_2 - beta_3_2_2_2;
            beta_3_2_2    = beta_3_2_2 + Cb0(1) * Cb0(1) * P * self.intcubmat(tau,D,D(1,1) + D(1,1),Gam2_11) * Cb0;
            beta_3_2_2    = beta_3_2_2 + Cb0(1) * Cb0(2) * P * self.intcubmat(tau,D,D(1,1) + D(2,2),Gam2_12) * Cb0;
            beta_3_2_2    = beta_3_2_2 + Cb0(1) * Cb0(3) * P * self.intcubmat(tau,D,D(1,1) + D(3,3),Gam2_13) * Cb0;

            beta_3_2_2    = beta_3_2_2 + Cb0(2) * Cb0(1) * P * self.intcubmat(tau,D,D(2,2) + D(1,1),Gam2_21) * Cb0;
            beta_3_2_2    = beta_3_2_2 + Cb0(2) * Cb0(2) * P * self.intcubmat(tau,D,D(2,2) + D(2,2),Gam2_22) * Cb0;
            beta_3_2_2    = beta_3_2_2 + Cb0(2) * Cb0(3) * P * self.intcubmat(tau,D,D(2,2) + D(3,3),Gam2_23) * Cb0;

            beta_3_2_2    = beta_3_2_2 + Cb0(3) * Cb0(1) * P * self.intcubmat(tau,D,D(3,3) + D(1,1),Gam2_31) * Cb0;
            beta_3_2_2    = beta_3_2_2 + Cb0(3) * Cb0(2) * P * self.intcubmat(tau,D,D(3,3) + D(2,2),Gam2_32) * Cb0;
            beta_3_2_2    = beta_3_2_2 + Cb0(3) * Cb0(3) * P * self.intcubmat(tau,D,D(3,3) + D(3,3),Gam2_33) * Cb0;
            %%% and the Part 2 sum
            beta_3_2      = beta_3_2_1 + beta_3_2_2;
            %%%% Part 3
            %%%% Part 3-1
            phib3_c      =  A_i * [(B3_1hc - B3_1hs * C0).'; (B3_3hc - B3_3hs * C0).'; (B3_2hc - B3_2hs * C0).'] * phib_c;
            beta_3_3_1_1 =  expm(tau * A) * phib3_c - phib3_c;
            beta_3_3_1_1 =  beta_3_3_1_1 + P * (self.intcarmat(tau, D, lamd_1) * B3_1_hc + self.intcarmat(tau, D, lamd_2) * B3_2_hc + self.intcarmat(tau,D,-kappa_2) * B3_3_hc) * P_i * phib_c;
            beta_3_3_1_1 = -beta_3_3_1_1;
            beta_3_3_1_2 =  P * (self.intcubmat(tau,D,0,B3_0_hc) + self.intcubmat(tau,D,lamd_1,B3_1_hc) + self.intcubmat(tau,D,lamd_2,B3_2_hc) + self.intcubmat(tau,D,-kappa_2,B3_3_hc))*P_i * phib_c;
            beta_3_3_1   =  beta_3_3_1_1 + beta_3_3_1_2;
            %%%% Part 3-2
            Pi_1        =  [0, phi_s(1,2)/(lamd_1 - lamd_2), phi_s(1,3)/(lamd_1 + kappa_2); phi_s(2,1)/(lamd_1 - lamd_2), 0, 0; phi_s(3,1)/(lamd_1 + kappa_2), 0, 0];
            Pi_2        =  [0, phi_s(1,2)/(lamd_2 - lamd_1), 0; phi_s(2,1)/(lamd_2 - lamd_1),0,phi_s(2,3)/(lamd_2 + kappa_2); 0, phi_s(3,2)/(lamd_2 + kappa_2), 0];
            Pi_3        = -[0, 0, phi_s(1,3)/(lamd_1 + kappa_2); 0, 0, phi_s(2,3)/(lamd_2 + kappa_2); phi_s(3,1)/(lamd_1 + kappa_2), phi_s(3,2)/(lamd_2 + kappa_2), 0];
            beta_3_3_2_1=  P * (self.intcarmat(tau,D,lamd_1) * B3_0_hc * Pi_1 +...
                phi_s(1,1) * self.int2carmat(tau,D,lamd_1) * B3_0_hc * E_1 +...
                self.intcarmat(tau,D,lamd_2) * B3_0_hc * Pi_2 +...
                phi_s(2,2) * self.int2carmat(tau,D,lamd_2) * B3_0_hc * E_2 +...
                self.intcarmat(tau,D,-kappa_2) * B3_0_hc * Pi_3 +...
                phi_s(3,3) * self.int2carmat(tau,D,-kappa_2) * B3_0_hc * E_3) * Cb0;
            Delta_1     =  P_i * [Cb0.' * Pi_1.'*Bb3_1hs; Cb0.' * Pi_1.' * Bb3_3hs; Cb0.' * Pi_1.' * Bb3_2hs];
            Delta_2     =  P_i * [Cb0.' * Pi_2.'*Bb3_1hs; Cb0.' * Pi_2.' * Bb3_3hs; Cb0.' * Pi_2.' * Bb3_2hs];
            Delta_3     =  P_i * [Cb0.' * Pi_3.'*Bb3_1hs; Cb0.' * Pi_3.' * Bb3_3hs; Cb0.' * Pi_3.' * Bb3_2hs];
            IV_1        =  P_i * [Cb0.' * E_1.'*Bb3_1hs;  Cb0.' * E_1.' * Bb3_3hs; Cb0.' * E_1.' * Bb3_2hs];
            IV_2        =  P_i * [Cb0.' * E_2.'*Bb3_1hs;  Cb0.' * E_2.' * Bb3_3hs; Cb0.' * E_2.' * Bb3_2hs];
            IV_3        =  P_i * [Cb0.' * E_3.'*Bb3_1hs;  Cb0.' * E_3.' * Bb3_3hs; Cb0.' * E_3.' * Bb3_2hs];
            beta_3_3_2_2=  P * (self.intcubmat(tau,D,lamd_1,Delta_1) + phi_s(1,1) * self.int2cubmat(tau,D,lamd_1,IV_1) + self.intcubmat(tau,D,lamd_2,Delta_2) + phi_s(2,2) * self.int2cubmat(tau,D,lamd_2,IV_2) + self.intcubmat(tau,D,-kappa_2,Delta_3) + phi_s(3,3) * self.int2cubmat(tau,D,-kappa_2,IV_3)) * Cb0;
            beta_3_3_2  =  beta_3_3_2_1 + beta_3_3_2_2;
            %%%% Part 3-3
            psi_1_2     = Cb0(1) * Cb0(2) * [Bt2_1q(2,1)/lamd_2; Bt2_3q(2,1)/lamd_1; 2 * Bt2_2q(2,1)/(lamd_1 + lamd_2 + kappa_2)];
            psi_1_3     = Cb0(1) * Cb0(3) * [0; 0; Bt2_2q(3,1)/lamd_1];
            psi_2_3     = Cb0(2) * Cb0(3) * [0; 0; Bt2_2q(3,2)/lamd_2];
            psi_2_2     = Cb0(2) * Cb0(2) * [Bt2_1q(2,2)/(2 * lamd_2 - lamd_1); 0; Bt2_2q(2,2)/(2 * lamd_2 + kappa_2)];
            psi_1_1     = Cb0(1) * Cb0(1) * [0; Bt2_3q(1,1)/(2 * lamd_1 - lamd_2); Bt2_2q(1,1)/(2 * lamd_1 + kappa_2)];
            psi_1       = Cb0(2) * [Cb0(1) * (Bt2_1q(2,1)/lamd_2) + Cb0(2) * Bt2_1q(2,2)/(2 * lamd_2 - lamd_1); 0; 0];
            psi_2       = Cb0(1) * [0; Cb0(1) * Bt2_3q(1,1)/(2 * lamd_1 - lamd_2) + Cb0(2) * (Bt2_3q(2,1)/lamd_1); 0];
            psi_3       = [0; 0; Cb0(1) * Cb0(1) * Bt2_2q(1,1)/(2 * lamd_1 + kappa_2) + Cb0(2) * Cb0(2) * Bt2_2q(2,2)/(2 * lamd_2 + kappa_2) + 2 * Cb0(1) * Cb0(2) * Bt2_2q(2,1)/(lamd_1 + lamd_2 + kappa_2) + Cb0(1) * Cb0(3) * Bt2_2q(3,1)/lamd_1 + Cb0(2) * Cb0(3) * Bt2_2q(3,2)/lamd_2];

            Gpsi_1_2    = P_i * [psi_1_2.' * Bb3_1hs; psi_1_2.' * Bb3_3hs; psi_1_2.' * Bb3_2hs];
            Gpsi_1_3    = P_i * [psi_1_3.' * Bb3_1hs; psi_1_3.' * Bb3_3hs; psi_1_3.' * Bb3_2hs];
            Gpsi_2_3    = P_i * [psi_2_3.' * Bb3_1hs; psi_2_3.' * Bb3_3hs; psi_2_3.' * Bb3_2hs];
            Gpsi_2_2    = P_i * [psi_2_2.' * Bb3_1hs; psi_2_2.' * Bb3_3hs; psi_2_2.' * Bb3_2hs];
            Gpsi_1_1    = P_i * [psi_1_1.' * Bb3_1hs; psi_1_1.' * Bb3_3hs; psi_1_1.' * Bb3_2hs];
            Gpsi_1      = P_i * [psi_1.' * Bb3_1hs; psi_1.' * Bb3_3hs; psi_1.' * Bb3_2hs];
            Gpsi_2      = P_i * [psi_2.' * Bb3_1hs; psi_2.' * Bb3_3hs; psi_2.' * Bb3_2hs];
            Gpsi_3      = P_i * [psi_3.' * Bb3_1hs; psi_3.' * Bb3_3hs; psi_3.' * Bb3_2hs];

            beta_3_3_3_1    = P * (self.intcarmat(tau,D,lamd_1+lamd_2) * B3_0_hc * psi_1_2 +...
                self.intcarmat(tau,D,lamd_1-kappa_2) * B3_0_hc * psi_1_3 +...
                self.intcarmat(tau,D,lamd_2-kappa_2) * B3_0_hc * psi_2_3 +...
                self.intcarmat(tau,D,lamd_2 + lamd_2) * B3_0_hc * psi_2_2 +...
                self.intcarmat(tau,D,lamd_1+lamd_1) * B3_0_hc * psi_1_1 -...
                self.intcarmat(tau,D,lamd_1) * B3_0_hc * psi_1 -...
                self.intcarmat(tau,D,lamd_2) * B3_0_hc * psi_2 -...
                self.intcarmat(tau,D,-kappa_2) * B3_0_hc * psi_3);
            beta_3_3_3_2    = P * (self.intcubmat(tau,D,lamd_1+lamd_2,Gpsi_1_2) +...
                self.intcubmat(tau,D,lamd_1-kappa_2,Gpsi_1_3) +...
                self.intcubmat(tau,D,lamd_2 - kappa_2,Gpsi_2_3) +...
                self.intcubmat(tau,D,lamd_2 + lamd_2,Gpsi_2_2) +...
                self.intcubmat(tau,D,lamd_1 + lamd_1,Gpsi_1_1) -...
                self.intcubmat(tau,D,lamd_1,Gpsi_1) -...
                self.intcubmat(tau,D,lamd_2,Gpsi_2) -...
                self.intcubmat(tau,D,-kappa_2,Gpsi_3)) * Cb0;
            beta_3_3_3      = beta_3_3_3_1 + beta_3_3_3_2;
            %%% and the Part 3 sum
            beta_3_3        = beta_3_3_1 + beta_3_3_2 + beta_3_3_3;
            %%%% the sum
            beta_3          = beta_3_1 + beta_3_2 + beta_3_3;
            %%%% alpha_3
            alpha_3         = Abalpha.' * beta_3;
            alpha_3         = alpha_3 + alphab3_c * tau;
            alpha_3         = alpha_3 + alphab3_s.' * int_beta_1;
            alpha_3         = alpha_3 + alphab3_hc.' * int_beta_2;
            alpha_3         = alpha_3 + C0.' * self.intsquaremat(P,P_i,D,tau,alphab3_qc) * C0 - 2 * C0.' * alphab3_qc * A_i * beta_1 + tau * C0.' * alphab3_qc * C0;%%int_produit des bata_1
            alpha_3         = alpha_3 + C0.' * diag([alphab3_1_qs.' * int_beta_1; alphab3_3_qs.' * int_beta_1; alphab3_2_qs.' * int_beta_1]) * C0;
            alpha_3         = alpha_3 - 2 * C0.' * [P(1,1) * alphab3_1_qs.' * self.intexplamd_beta_1(tau,D(1,1),A,C0),P(1,2) * alphab3_1_qs.' * self.intexplamd_beta_1(tau,D(2,2),A,C0),P(1,3) * alphab3_1_qs.' * self.intexplamd_beta_1(tau,D(3,3),A,C0);...
                P(2,1) * alphab3_3_qs.' * self.intexplamd_beta_1(tau,D(1,1),A,C0),P(2,2) * alphab3_3_qs.' * self.intexplamd_beta_1(tau,D(2,2),A,C0),P(2,3) * alphab3_3_qs.' * self.intexplamd_beta_1(tau,D(3,3),A,C0);...
                P(3,1) * alphab3_2_qs.' * self.intexplamd_beta_1(tau,D(1,1),A,C0),P(3,2) * alphab3_2_qs.' * self.intexplamd_beta_1(tau,D(2,2),A,C0),P(3,3) * alphab3_2_qs.' * self.intexplamd_beta_1(tau,D(3,3),A,C0)] * P_i * C0;
            P_hat           = zeros(3,3);
            P_hat(1,1)      = (P(1,1) * P(1,1) * alphab3_1_qs.' + P(2,1) * P(2,1) * alphab3_3_qs.' + P(3,1) * P(3,1) * alphab3_2_qs.') * self.intexplamd_beta_1(tau,D(1,1) + D(1,1),A,C0);
            P_hat(1,2)      = (P(1,1) * P(1,2) * alphab3_1_qs.' + P(2,1) * P(2,2) * alphab3_3_qs.' + P(3,1) * P(3,2) * alphab3_2_qs.') * self.intexplamd_beta_1(tau,D(1,1) + D(2,2),A,C0);
            P_hat(1,3)      = (P(1,1) * P(1,3) * alphab3_1_qs.' + P(2,1) * P(2,3) * alphab3_3_qs.' + P(3,1) * P(3,3) * alphab3_2_qs.') * self.intexplamd_beta_1(tau,D(1,1) + D(3,3),A,C0);

            P_hat(2,1)      = (P(1,2) * P(1,1) * alphab3_1_qs.' + P(2,2) * P(2,1) * alphab3_3_qs.' + P(3,2) * P(3,1) * alphab3_2_qs.') * self.intexplamd_beta_1(tau,D(2,2) + D(1,1),A,C0);
            P_hat(2,2)      = (P(1,2) * P(1,2) * alphab3_1_qs.' + P(2,2) * P(2,2) * alphab3_3_qs.' + P(3,2) * P(3,2) * alphab3_2_qs.') * self.intexplamd_beta_1(tau,D(2,2) + D(2,2),A,C0);
            P_hat(2,3)      = (P(1,2) * P(1,3) * alphab3_1_qs.' + P(2,2) * P(2,3) * alphab3_3_qs.' + P(3,2) * P(3,3) * alphab3_2_qs.') * self.intexplamd_beta_1(tau,D(2,2) + D(3,3),A,C0);

            P_hat(3,1)      = (P(1,3) * P(1,1) * alphab3_1_qs.' + P(2,3) * P(2,1) * alphab3_3_qs.' + P(3,3) * P(3,1) * alphab3_2_qs.') * self.intexplamd_beta_1(tau,D(3,3) + D(1,1),A,C0);
            P_hat(3,2)      = (P(1,3) * P(1,2) * alphab3_1_qs.' + P(2,3) * P(2,2) * alphab3_3_qs.' + P(3,3) * P(3,2) * alphab3_2_qs.') * self.intexplamd_beta_1(tau,D(3,3) + D(2,2),A,C0);
            P_hat(3,3)      = (P(1,3) * P(1,3) * alphab3_1_qs.' + P(2,3) * P(2,3) * alphab3_3_qs.' + P(3,3) * P(3,3) * alphab3_2_qs.') * self.intexplamd_beta_1(tau,D(3,3) + D(3,3),A,C0);

            alpha_3         = alpha_3 + C0.' * P_i.' * P_hat * P_i * C0;
            alpha_3         = alpha_3 - C0.' * alphab3_hs * int_beta_2;
            alpha_3         = alpha_3 + C0.' * (self.intsquaremat(P, P_i, D, tau, alphab3_hs) - A_i.' * (expm(tau * A.') - eye(3)) * alphab3_hs) * phib_c;
            alpha_3         = alpha_3 + C0.' * (self.Am_i(A.', lamd_1) * (expm(tau * (A.' + lamd_1 * eye(3))) - eye(3)) * alphab3_hs * P * Pi_1 * Cb0 + phi_s(1,1) * (tau * self.Am_i(A.',lamd_1) * expm(tau * (A.' + lamd_1 * eye(3))) - self.Am_i(A.',lamd_1) * self.Am_i(A.',lamd_1) * (expm(tau * (A.' + lamd_1 * eye(3))) - eye(3))) * alphab3_hs * P * E_1 * Cb0);
            alpha_3         = alpha_3 + C0.' * (self.Am_i(A.', lamd_2) * (expm(tau * (A.' + lamd_2 * eye(3))) - eye(3)) * alphab3_hs * P * Pi_2 * Cb0 + phi_s(2,2) * (tau * self.Am_i(A.',lamd_2) * expm(tau * (A.' + lamd_2 * eye(3))) - self.Am_i(A.',lamd_2) * self.Am_i(A.',lamd_2) * (expm(tau * (A.' + lamd_2 * eye(3))) - eye(3))) * alphab3_hs * P * E_2 * Cb0);
            alpha_3         = alpha_3 + C0.' * (self.Am_i(A.', lamd_3) * (expm(tau * (A.' + lamd_3 * eye(3))) - eye(3)) * alphab3_hs * P * Pi_3 * Cb0 + phi_s(3,3) * (tau * self.Am_i(A.',lamd_3) * expm(tau * (A.' + lamd_3 * eye(3))) - self.Am_i(A.',lamd_3) * self.Am_i(A.',lamd_3) * (expm(tau * (A.' + lamd_3 * eye(3))) - eye(3))) * alphab3_hs * P * E_3 * Cb0);
            alpha_3         = alpha_3 - C0.' * self.Am_i(A.', lamd_1) * (expm(tau * (A.' + lamd_1 * eye(3))) - eye(3)) * alphab3_hs * P * psi_1 - C0.' * self.Am_i(A.',lamd_2) * (expm(tau * (A.' + lamd_2 * eye(3))) - eye(3)) * alphab3_hs * P * psi_2 - C0.' * self.Am_i(A.',lamd_3) * (expm(tau * (A.' + lamd_3 * eye(3))) - eye(3)) * alphab3_hs * P * psi_3;
            alpha_3         = alpha_3 + C0.' * (self.Am_i(A.', lamd_1 + lamd_1) * (expm(tau * (A.' + (lamd_1 + lamd_1) * eye(3)))-eye(3)) * alphab3_hs * P * psi_1_1 +...
                self.Am_i(A.',lamd_1 + lamd_2) * (expm(tau * (A.' + (lamd_1 + lamd_2) * eye(3))) - eye(3)) * alphab3_hs * P * psi_1_2 +...
                self.Am_i(A.',lamd_1 + lamd_3) * (expm(tau * (A.' + (lamd_1 + lamd_3) * eye(3))) - eye(3)) * alphab3_hs * P * psi_1_3);
            alpha_3         = alpha_3 + C0.' * (self.Am_i(A.', lamd_2 + lamd_2) * (expm(tau * (A.' + (lamd_2 + lamd_2) * eye(3)))-eye(3)) * alphab3_hs * P * psi_2_2 + ...
                self.Am_i(A.',lamd_2 + lamd_3) * (expm(tau * (A.' + (lamd_2 + lamd_3) * eye(3))) - eye(3)) * alphab3_hs * P * psi_2_3);

            %%%%%%% compute the integrant for alpha_4 and _beta_4;
            B4_1c=c1_m*Theta_nc.q0q0q0q0+c1_p*Theta_p.q0q0q0q0;
            B4_3c=c3_m*Theta_nc.q0q0q0q0+c3_p*Theta_p.q0q0q0q0;
            B4_2c=c2_m*Theta_nc.q0q0q0q0+c2_p*Theta_p.q0q0q0q0;
            alpha4_c=c0_m*Theta_nc.q0q0q0q0+c0_p*Theta_p.q0q0q0q0;

            %%%
            B4_1s=[4*c1_m*Theta_nc.q0q0q0q1;4*c1_m*Theta_nc.q0q0q0q3;0];
            B4_3s=[4*c3_m*Theta_nc.q0q0q0q1;4*c3_m*Theta_nc.q0q0q0q3;0];
            B4_2s=[4*c2_m*Theta_nc.q0q0q0q1;4*c2_m*Theta_nc.q0q0q0q3;0];
            alpha4_s=[4*c0_m*Theta_nc.q0q0q0q1;4*c0_m*Theta_nc.q0q0q0q3;0];

            %%%
            B4_1qc=[6*c1_m*Theta_nc.q0q0q1q1, 6*c1_m*Theta_nc.q0q0q1q3, 0; 6*c1_m*Theta_nc.q0q0q1q3, 6*c1_m*Theta_nc.q0q0q3q3, 0; 0, 0, 0];
            B4_3qc=[6*c3_m*Theta_nc.q0q0q1q1, 6*c3_m*Theta_nc.q0q0q1q3, 0; 6*c3_m*Theta_nc.q0q0q1q3, 6*c3_m*Theta_nc.q0q0q3q3, 0; 0, 0, 0];
            B4_2qc=[6*c2_m*Theta_nc.q0q0q1q1, 6*c2_m*Theta_nc.q0q0q1q3, 0; 6*c2_m*Theta_nc.q0q0q1q3, 6*c2_m*Theta_nc.q0q0q3q3, 0; 0, 0, 0];
            alpha4_qc=[6*c0_m*Theta_nc.q0q0q1q1, 6*c0_m*Theta_nc.q0q0q1q3, 0; 6*c0_m*Theta_nc.q0q0q1q3, 6*c0_m*Theta_nc.q0q0q3q3, 0; 0, 0, 0];

            %%%
            B4_1_1qs=[4*c1_m*Theta_nc.q0q1q1q1;12*c1_m*Theta_nc.q0q1q1q3;0];
            B4_3_1qs=[12*c1_m*Theta_nc.q0q1q3q3;4*c1_m*Theta_nc.q0q3q3q3;0];

            %%%
            alpha4_1_qs=[4*c0_m*Theta_nc.q0q1q1q1;12*c0_m*Theta_nc.q0q1q1q3;0];
            alpha4_3_qs=[12*c0_m*Theta_nc.q0q1q3q3;4*c0_m*Theta_nc.q0q3q3q3;0];

            %%%
            B4_1_3qs=[4*c3_m*Theta_nc.q0q1q1q1;12*c3_m*Theta_nc.q0q1q1q3;0];
            B4_3_3qs=[12*c3_m*Theta_nc.q0q1q3q3;4*c3_m*Theta_nc.q0q3q3q3;0];

            %%%
            B4_1_2qs=[4*c2_m*Theta_nc.q0q1q1q1;12*c2_m*Theta_nc.q0q1q1q3;0];
            B4_3_2qs=[12*c2_m*Theta_nc.q0q1q3q3;4*c2_m*Theta_nc.q0q3q3q3;0];

            %%%
            B4_1_1qq=[c1_m*Theta_nc.q1q1q1q1, 2*c1_m*Theta_nc.q1q1q1q3,0;2*c1_m*Theta_nc.q1q1q1q3, 3*c1_m*Theta_nc.q1q1q3q3, 0; 0, 0, 0];
            B4_3_1qq=[3*c1_m*Theta_nc.q1q1q3q3, 2*c1_m*Theta_nc.q1q3q3q3,0;2*c1_m*Theta_nc.q1q3q3q3, c1_m*Theta_nc.q3q3q3q3+c1_m*Theta_ni.q3q3q3q3, 0; 0, 0, 0];

            %%%
            alpha4_1_qq=[c0_m*Theta_nc.q1q1q1q1, 2*c0_m*Theta_nc.q1q1q1q3,0;2*c0_m*Theta_nc.q1q1q1q3, 3*c0_m*Theta_nc.q1q1q3q3, 0; 0, 0, 0];
            alpha4_3_qq=[3*c0_m*Theta_nc.q1q1q3q3, 2*c0_m*Theta_nc.q1q3q3q3,0;2*c0_m*Theta_nc.q1q3q3q3, c0_m*Theta_nc.q3q3q3q3+c0_m*Theta_ni.q3q3q3q3, 0; 0, 0, 0];

            %%%
            B4_1_3qq=[c3_m*Theta_nc.q1q1q1q1, 2*c3_m*Theta_nc.q1q1q1q3,0;2*c3_m*Theta_nc.q1q1q1q3, 3*c3_m*Theta_nc.q1q1q3q3, 0; 0, 0, 0];
            B4_3_3qq=[3*c3_m*Theta_nc.q1q1q3q3, 2*c3_m*Theta_nc.q1q3q3q3,0;2*c3_m*Theta_nc.q1q3q3q3, c3_m*Theta_nc.q3q3q3q3+c3_m*Theta_ni.q3q3q3q3, 0; 0, 0, 0];

            %%%
            B4_1_2qq=[c2_m*Theta_nc.q1q1q1q1, 2*c2_m*Theta_nc.q1q1q1q3,0;2*c2_m*Theta_nc.q1q1q1q3, 3*c2_m*Theta_nc.q1q1q3q3, 0; 0, 0, 0];
            B4_3_2qq=[3*c2_m*Theta_nc.q1q1q3q3, 2*c2_m*Theta_nc.q1q3q3q3,0;2*c2_m*Theta_nc.q1q3q3q3, c2_m*Theta_nc.q3q3q3q3+c2_m*Theta_ni.q3q3q3q3, 0; 0, 0, 0];

            %%%
            B4_1_1qsh=[6*c1_m*Theta_nc.q1q1q1;6*c1_m*Theta_nc.q1q1q3;0];
            B4_3_1qsh=[6*c1_m*Theta_nc.q1q3q3;6*c1_m*Theta_nc.q3q3q3+6*c1_m*Theta_ni.q3q3q3;0];

            %%%
            alpha4_1_qsh=[6*c0_m*Theta_nc.q1q1q1;6*c0_m*Theta_nc.q1q1q3;0];
            alpha4_3_qsh=[6*c0_m*Theta_nc.q1q3q3;6*c0_m*Theta_nc.q3q3q3+6*c0_m*Theta_ni.q3q3q3;0];

            %%%
            B4_1_3qsh=[6*c3_m*Theta_nc.q1q1q1;6*c3_m*Theta_nc.q1q1q3;0];
            B4_3_3qsh=[6*c3_m*Theta_nc.q1q3q3;6*c3_m*Theta_nc.q3q3q3+6*c3_m*Theta_ni.q3q3q3;0];

            %%%
            B4_1_2qsh=[6*c2_m*Theta_nc.q1q1q1;6*c2_m*Theta_nc.q1q1q3;0];
            B4_3_2qsh=[6*c2_m*Theta_nc.q1q3q3;6*c2_m*Theta_nc.q3q3q3+6*c2_m*Theta_ni.q3q3q3;0];

            %%%
            B4_1_2_1qsh=[6*c1_m*Theta_nc.q1q1q3;6*c1_m*Theta_nc.q1q3q3;0];
            B4_1_2_3qsh=[6*c3_m*Theta_nc.q1q1q3;6*c3_m*Theta_nc.q1q3q3;0];
            B4_1_2_2qsh=[6*c2_m*Theta_nc.q1q1q3;6*c2_m*Theta_nc.q1q3q3;0];
            alpha4_1_2_qsh=[6*c0_m*Theta_nc.q1q1q3;6*c0_m*Theta_nc.q1q3q3;0];

            %%%
            B4_1qs      = [B4_1_1qs.' * beta_1 + beta_1.' * B4_1_1qq * beta_1 + B4_1_1qsh.' * beta_2, B4_1_2_1qsh.' * beta_2, 0;...
                B4_1_2_1qsh.' * beta_2, B4_3_1qs.' * beta_1+beta_1.' * B4_3_1qq * beta_1 + B4_3_1qsh.' * beta_2, 0;...
                0, 0, 0];
            B4_3qs      = [B4_1_3qs.' * beta_1 + beta_1.'*B4_1_3qq * beta_1+B4_1_3qsh.' * beta_2, B4_1_2_3qsh.' * beta_2, 0;...
                B4_1_2_3qsh.' * beta_2, B4_3_3qs.' * beta_1+beta_1.' * B4_3_3qq * beta_1 + B4_3_3qsh.' * beta_2, 0;...
                0,0,0];
            B4_2qs      = [B4_1_2qs.' * beta_1 + beta_1.' * B4_1_2qq * beta_1 + B4_1_2qsh.' * beta_2, B4_1_2_2qsh.' * beta_2, 0;...
                B4_1_2_2qsh.' * beta_2, B4_3_2qs.' * beta_1 + beta_1.' * B4_3_2qq * beta_1 + B4_3_2qsh.' * beta_2, 0;...
                0,0,0];
            alpha4_qs   = [alpha4_1_qs.' * beta_1 + beta_1.' * alpha4_1_qq * beta_1 + alpha4_1_qsh.' * beta_2, alpha4_1_2_qsh.' * beta_2, 0;...
                alpha4_1_2_qsh.' * beta_2, alpha4_3_qs.' * beta_1+beta_1.' * alpha4_3_qq * beta_1 + alpha4_3_qsh.' * beta_2, 0;...
                0, 0, 0];

            %%%
            B4_1q       = B4_1qc + B4_1qs;
            B4_3q       = B4_3qc + B4_3qs;
            B4_2q       = B4_2qc + B4_2qs;
            alpha4_q    = alpha4_qc + alpha4_qs;

            %%%
            B4_1hc      = [ 6 * c1_m * Theta_nc.q0q0q1; 6 * c1_m * Theta_nc.q0q0q3; 0];
            B4_3hc      = [ 6 * c3_m * Theta_nc.q0q0q1; 6 * c3_m * Theta_nc.q0q0q3; 0];
            B4_2hc      = [ 6 * c2_m * Theta_nc.q0q0q1; 6 * c2_m * Theta_nc.q0q0q3; 0];
            alpha4_hc   = [ 6 * c0_m * Theta_nc.q0q0q1; 6 * c0_m * Theta_nc.q0q0q3; 0];

            %%%
            B4_1hs      = [ 12 * c1_m * Theta_nc.q0q1q1, 12 * c1_m * Theta_nc.q0q1q3, 0; 12 * c1_m * Theta_nc.q0q1q3, 12 * c1_m * Theta_nc.q0q3q3, 0; 0, 0, 0];
            B4_3hs      = [ 12 * c3_m * Theta_nc.q0q1q1, 12 * c3_m * Theta_nc.q0q1q3, 0; 12 * c3_m * Theta_nc.q0q1q3, 12 * c3_m * Theta_nc.q0q3q3, 0; 0, 0, 0];
            B4_2hs      = [ 12 * c2_m * Theta_nc.q0q1q1, 12 * c2_m * Theta_nc.q0q1q3, 0; 12 * c2_m * Theta_nc.q0q1q3, 12 * c2_m * Theta_nc.q0q3q3, 0; 0, 0, 0];
            alpha4_hs   = [ 12 * c0_m * Theta_nc.q0q1q1, 12 * c0_m * Theta_nc.q0q1q3, 0; 12 * c0_m * Theta_nc.q0q1q3, 12 * c0_m * Theta_nc.q0q3q3, 0; 0, 0, 0];

            %%%
            B4_1h=B4_1hc+B4_1hs*beta_1;
            B4_3h=B4_3hc+B4_3hs*beta_1;
            B4_2h=B4_2hc+B4_2hs*beta_1;
            alpha4_h=alpha4_hc+alpha4_hs*beta_1;

            %%%
            B4_1hq=[3*sigma_1^2+3*c1_m*Theta_nc.q1q1, 3*c1_m*Theta_nc.q1q3, 0; 3*c1_m*Theta_nc.q1q3, 3*c1_m*Theta_nc.q3q3+3*c1_m*Theta_ni.q3q3, 0; 0, 0, 0];
            B4_3hq=[3*c3_m*Theta_nc.q1q1, 3*c3_m*Theta_nc.q1q3, 0; 3*c3_m*Theta_nc.q1q3, 3*c3_m*Theta_nc.q3q3+3*c3_m*Theta_ni.q3q3, 0; 0, 0, 0];
            B4_2hq=[3*c2_m*Theta_nc.q1q1, 3*c2_m*Theta_nc.q1q3, 0; 3*c2_m*Theta_nc.q1q3, 3*c2_m*Theta_nc.q3q3+3*c2_m*Theta_ni.q3q3, 0; 0, 0, 3*sigma_2^2];
            alpha4_hq=[3*c0_m*Theta_nc.q1q1, 3*c0_m*Theta_nc.q1q3, 0; 3*c0_m*Theta_nc.q1q3, 3*c0_m*Theta_nc.q3q3+3*c0_m*Theta_ni.q3q3, 0; 0, 0, 0];

            %%%
            B4_1lc=[4*sigma_1*rho_1+4*c1_m*Theta_nc.q0q1;4*c1_m*Theta_nc.q0q3;0];
            B4_3lc=[4*c3_m*Theta_nc.q0q1;4*c3_m*Theta_nc.q0q3;0];
            B4_2lc=[4*c2_m*Theta_nc.q0q1;4*c2_m*Theta_nc.q0q3;4*sigma_2*rho_2];
            alpha4_lc=[4*c0_m*Theta_nc.q0q1;4*c0_m*Theta_nc.q0q3;0];

            %%%
            B4_1ls=[4*sigma_1^2+4*c1_m*Theta_nc.q1q1, 4*c1_m*Theta_nc.q1q3, 0; 4*c1_m*Theta_nc.q1q3, 4*c1_m*Theta_nc.q3q3+4*c1_m*Theta_ni.q3q3, 0; 0, 0, 0];
            B4_3ls=[4*c3_m*Theta_nc.q1q1, 4*c3_m*Theta_nc.q1q3, 0; 4*c3_m*Theta_nc.q1q3, 4*c3_m*Theta_nc.q3q3+4*c3_m*Theta_ni.q3q3, 0; 0, 0, 0];
            B4_2ls=[4*c2_m*Theta_nc.q1q1, 4*c2_m*Theta_nc.q1q3, 0; 4*c2_m*Theta_nc.q1q3, 4*c2_m*Theta_nc.q3q3+4*c2_m*Theta_ni.q3q3, 0; 0, 0, 4*sigma_2^2];
            alpha4_ls=[4*c0_m*Theta_nc.q1q1, 4*c0_m*Theta_nc.q1q3, 0; 4*c0_m*Theta_nc.q1q3, 4*c0_m*Theta_nc.q3q3+4*c0_m*Theta_ni.q3q3, 0; 0, 0, 0];

            %%%
            B4_1l=B4_1lc+B4_1ls*beta_1;
            B4_3l=B4_3lc+B4_3ls*beta_1;
            B4_2l=B4_2lc+B4_2ls*beta_1;
            alpha4_l=alpha4_lc+alpha4_ls*beta_1;

            %%%
            B4_1=B4_1c+B4_1s.'*beta_1+beta_1.'*B4_1q*beta_1+B4_1h.'*beta_2+beta_2.'*B4_1hq*beta_2+B4_1l.'*beta_3;
            B4_3=B4_3c+B4_3s.'*beta_1+beta_1.'*B4_3q*beta_1+B4_3h.'*beta_2+beta_2.'*B4_3hq*beta_2+B4_3l.'*beta_3;
            B4_2=B4_2c+B4_2s.'*beta_1+beta_1.'*B4_2q*beta_1+B4_2h.'*beta_2+beta_2.'*B4_2hq*beta_2+B4_2l.'*beta_3;
            B4=[B4_1;B4_3;B4_2];
            Alpha4=alpha4_c+alpha4_s.'*beta_1+beta_1.'*alpha4_q*beta_1+alpha4_h.'*beta_2+beta_2.'*alpha4_hq*beta_2+alpha4_l.'*beta_3;
        end % end AFT_RNM_Coeff

        function [f] = AFT_KURT_Coeff(self, tau)
            % following the Appendix Feunou and Okou 2016
            f=integral(@(s) self.intc(s,tau), 0, tau, 'AbsTol', 1e-05, 'RelTol', 1e-05, 'ArrayValued', true);
        end % end AFT_KURT_Coeff

        function f=intc(self, s, tau)
            % following the Appendix Feunou and Okou 2016

            % copy of parameters (in a structure)
            pv      = self.getPV();

            eta      = pv.eta;       c0_m    = pv.cn0;      c3_p    = pv.cp3;
            rho_1    = pv.rho1;      vb_1    = pv.nu1;      kappa_1 = pv.kappa1;    sigma_1 = pv.sigma1;
            rho_2    = pv.rho2;      vb_2    = pv.nu2;      kappa_2 = pv.kappa2;    sigma_2 = pv.sigma2;
            mu_3     = pv.mu3;       kappa_3 = pv.kappa3;   rho_3   = pv.rho3;
            c0_p     = pv.cp0;       c1_m    = pv.cn1;      c1_p    = pv.cp1;       c2_m    = pv.cn2;   c2_p    = pv.cp2;   c3_m    = pv.cn3;
            lamd_m   = pv.lambdan;   lamd_p  = pv.lambdap;  mu_1    = pv.mu1;
            Theta_nc = self.Theta_nc_Derivative(mu_1,mu_3,rho_3,lamd_m);
            Theta_ni = self.Theta_ni_Derivative(mu_3,rho_3,lamd_m);

            A       = [-kappa_1 + c1_m * Theta_nc.q1, c1_m * Theta_nc.q3 + c1_m * Theta_ni.q3, 0;...
                c3_m * Theta_nc.q1, -kappa_3 + c3_m * Theta_nc.q3 + c3_m * Theta_ni.q3, 0;...
                c2_m * Theta_nc.q1, c2_m * Theta_nc.q3 + c2_m * Theta_ni.q3, -kappa_2];
            Aalpha  = [kappa_1 * vb_1 + c0_m * Theta_nc.q1;...
                c0_m * Theta_nc.q3 + c0_m * Theta_ni.q3;...
                kappa_2 * vb_2];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            nb      =numel(s);
            f       =zeros(4,nb);
            for j = 1:nb
                [~,~,~,~,B4,Alpha4] = self.AFT_RNM_Coeff(s(j));
                f(1,j)      =  Alpha4 + (Aalpha.'/A) * (expm((tau - s(j)) * A) - eye(3)) * B4;
                f(2:end,j)  =  expm((tau - s(j)) * A) * B4;
            end
        end % end intc

        function Theta_ni = Theta_ni_Derivative(self, mu_3, rho_3, lamd_m)
            %%% This function computes the derivatives of Theta_ni, following the Appendix Feunou and Okou 2016

            % first-order
            Theta_ni.q3         = (2/lamd_m^2) * (rho_3 * mu_3);
            % second-order
            Theta_ni.q3q3       = (24/lamd_m^4) * (rho_3 * mu_3)^2;
            % third-order
            Theta_ni.q3q3q3     = (720/lamd_m^6) * (rho_3 * mu_3)^3;
            % fourth-order
            Theta_ni.q3q3q3q3   = (40320/lamd_m^8) * (rho_3 * mu_3)^4;
        end % end Theta_ni_Derivative

        function [Theta_nc] = Theta_nc_Derivative(self, mu_1, mu_3, rho_3, lamd_m)
            %%% This function computes the derivatives of Theta_nc, following the Appendix Feunou and Okou 2016

            % first-order
            Theta_nc.q0         = -1/lamd_m;                Theta_nc.q1     =  (2/lamd_m^2) * mu_1;                   Theta_nc.q3       =  (2/lamd_m^2) * (1-rho_3) * mu_3;
            % second-order
            Theta_nc.q0q0       =   2/lamd_m^2;             Theta_nc.q1q1   =  (24/lamd_m^4) * mu_1^2;                Theta_nc.q3q3     =  (24/lamd_m^4) * ((1-rho_3) * mu_3)^2;
            Theta_nc.q0q1       = -(6/lamd_m^3) * mu_1;     Theta_nc.q0q3   = -(6/lamd_m^3) * (1-rho_3) * mu_3;       Theta_nc.q1q3     =  (24/lamd_m^4) * mu_1 * (1-rho_3)*mu_3;
            % third-order
            Theta_nc.q0q0q0     = -(6/lamd_m^3);            Theta_nc.q1q1q1 =  (720/lamd_m^6) * mu_1^3;               Theta_nc.q3q3q3   =  (720/lamd_m^6) * ((1-rho_3) * mu_3)^3;
            Theta_nc.q0q0q1     =  (24/lamd_m^4) * mu_1;    Theta_nc.q0q0q3 =  (24/lamd_m^4) * (1-rho_3) * mu_3;      Theta_nc.q0q1q1   = -(120/lamd_m^5) * mu_1^2;
            Theta_nc.q0q1q3     = -(120/lamd_m^5) * mu_1 * (1-rho_3)*mu_3;          Theta_nc.q0q3q3 = -(120/lamd_m^5) * ((1-rho_3) * mu_3)^2;                 Theta_nc.q1q1q3       =  (720/lamd_m^6) * (mu_1^2) * ((1-rho_3) * mu_3);
            Theta_nc.q1q3q3     =  (720/lamd_m^6) * mu_1 * ((1-rho_3)*mu_3)^2;
            % fourth-order
            Theta_nc.q0q0q0q0   =   24/lamd_m^4;            Theta_nc.q1q1q1q1 =  (40320/lamd_m^8) * mu_1^4;                                                   Theta_nc.q3q3q3q3     =  (40320/lamd_m^8) * ((1-rho_3) * mu_3)^4;
            Theta_nc.q0q0q0q1   = -(120/lamd_m^5) * mu_1;   Theta_nc.q0q0q0q3 = -(120/lamd_m^5) * ((1-rho_3) * mu_3);                                         Theta_nc.q0q0q1q1     =  (720/lamd_m^6) * mu_1^2;
            Theta_nc.q0q0q1q3   =  (720/lamd_m^6) * mu_1 * ((1-rho_3)*mu_3);        Theta_nc.q0q0q3q3   =  (720/lamd_m^6) * ((1-rho_3) * mu_3)^2;             Theta_nc.q0q1q1q1     = -(5040/lamd_m^7) * mu_1^3;
            Theta_nc.q0q1q1q3   = -(5040/lamd_m^7) * (mu_1^2) * ((1-rho_3)*mu_3);   Theta_nc.q0q1q3q3   = -(5040/lamd_m^7) * (mu_1) * ((1-rho_3) * mu_3)^2;   Theta_nc.q0q3q3q3     = -(5040/lamd_m^7) * ((1 - rho_3) * mu_3)^3;
            Theta_nc.q1q1q1q3   =  (40320/lamd_m^8) * (mu_1^3) * ((1-rho_3)*mu_3);  Theta_nc.q1q1q3q3   =  (40320/lamd_m^8) * (mu_1^2) * ((1-rho_3) * mu_3)^2;Theta_nc.q1q3q3q3     =  (40320/lamd_m^8)*(mu_1) * ((1-rho_3) * mu_3)^3;

        end % end Theta_nc_Derivative

        function Theta_p = Theta_p_Derivative(self, lamd_p)
            %%% This function computes the derivatives of Theta_p, following the Appendix Feunou and Okou 2016
            % first-order
            Theta_p.q0          = 1/lamd_p;
            % second-order
            Theta_p.q0q0        = 2/lamd_p^2;
            % third-order
            Theta_p.q0q0q0      = 6/lamd_p^3;
            % fourth-order
            Theta_p.q0q0q0q0    = 24/lamd_p^4;
        end % end Theta_p_Derivative

        function R = intcubmat(self, tau, D, lamd, Gam)
            % following the Appendix Feunou and Okou 2016
            lamd_1  = D(1,1);       lamd_2 = D(2,2);      lamd_3    = D(3,3);
            R       = [Gam(1,1) * self.dexp(tau,lamd_1 + lamd,lamd_1), Gam(1,2) * self.dexp(tau,lamd_2 + lamd,lamd_1), Gam(1,3) * self.dexp(tau,lamd_3 + lamd,lamd_1);...
                Gam(2,1) * self.dexp(tau,lamd_1 + lamd,lamd_2), Gam(2,2) * self.dexp(tau,lamd_2 + lamd,lamd_2), Gam(2,3) * self.dexp(tau,lamd_3 + lamd,lamd_2);...
                Gam(3,1) * self.dexp(tau, lamd_1 + lamd, lamd_3), Gam(3,2) * self.dexp(tau,lamd_2 + lamd, lamd_3), Gam(3,3) * self.dexp(tau,lamd_3 + lamd,lamd_3)];
        end % end intcubmat

        function R  = dexp(self, tau, lamd_1, lamd_2)
            % following the Appendix Feunou and Okou 2016
            if lamd_1 == lamd_2
                R       = tau * exp(tau * lamd_1);
            else
                R       = (exp(tau * lamd_1) - exp(tau * lamd_2))/(lamd_1 - lamd_2);
            end
        end % end dexp

        function beta_2_3 = SecondMom3(self, tau, Bt2_1q, Bt2_3q, Bt2_2q, D, Cb0)
            % following the Appendix Feunou and Okou 2016
            lamd_1  = D(1,1);       lamd_2  = D(2,2);       kappa_2  =  -D(3,3);
            beta_2_3    = zeros(3,1);
            beta_2_3(1) = Cb0(1) * Cb0(2) * Bt2_1q(2,1) * ((exp(tau * (lamd_1 + lamd_2)) - exp(tau * lamd_1))/lamd_2) + Cb0(2) * Cb0(2) * Bt2_1q(2,2) * ((exp(tau * (lamd_2 + lamd_2)) - exp(tau * lamd_1))/(2 * lamd_2 - lamd_1));
            beta_2_3(2) = Cb0(1) * Cb0(1) * Bt2_3q(1,1) * ((exp(tau * (lamd_1 + lamd_1)) - exp(tau * lamd_2))/(2 * lamd_1 - lamd_2)) + Cb0(1) * Cb0(2) * Bt2_3q(2,1) * ((exp(tau * (lamd_1 + lamd_2)) - exp(tau * lamd_2))/lamd_1);
            beta_2_3(3) = Cb0(1) * Cb0(1) * Bt2_2q(1,1) * ((exp(tau * (lamd_1 + lamd_1)) - exp(-tau * kappa_2))/(2 * lamd_1 + kappa_2)) + Cb0(2) * Cb0(2) * Bt2_2q(2,2) * ((exp(tau * (lamd_2 + lamd_2)) - exp( -tau * kappa_2))/(2 * lamd_2 + kappa_2))...
                + 2 * Cb0(1) * Cb0(2) * Bt2_2q(2,1) * ((exp(tau * (lamd_1 + lamd_2)) - exp( -tau * kappa_2))/(lamd_1 + lamd_2 + kappa_2)) + Cb0(1) * Cb0(3) * Bt2_2q(3,1) * ((exp(tau * (lamd_1 - kappa_2)) - exp( -tau * kappa_2))/lamd_1)...
                + Cb0(2) * Cb0(3) * Bt2_2q(3,2) * ((exp(tau * (lamd_2 - kappa_2)) - exp( -tau * kappa_2))/lamd_2);
        end % end SecondMom3

        function R = intsquaremat(self, P, P_i, D, tau, Gam)
            % following the Appendix Feunou and Okou 2016
            lamd_1  = D(1,1);       lamd_2  = D(2,2);       lamd_3 = D(3,3);
            Gamb    = P.' * Gam * P;
            R       = [Gamb(1,1) * ((exp(tau * (lamd_1 + lamd_1)) - 1)/(lamd_1 + lamd_1)), Gamb(1,2) * ((exp(tau * (lamd_1 + lamd_2)) - 1)/(lamd_1 + lamd_2)), Gamb(1,3) * ((exp(tau * (lamd_1 + lamd_3)) - 1)/(lamd_1 + lamd_3));...
                Gamb(2,1) * ((exp(tau * (lamd_2 + lamd_1)) - 1)/(lamd_2 + lamd_1)), Gamb(2,2) * ((exp(tau * (lamd_2 + lamd_2)) - 1)/(lamd_2 + lamd_2)), Gamb(2,3) * ((exp(tau * (lamd_2 + lamd_3)) - 1)/(lamd_2 + lamd_3));...
                Gamb(3,1) * ((exp(tau * (lamd_3 + lamd_1)) - 1)/(lamd_3 + lamd_1)), Gamb(3,2) * ((exp(tau * (lamd_3 + lamd_2)) - 1)/(lamd_3 + lamd_2)), Gamb(3,3) * ((exp(tau * (lamd_3 + lamd_3)) - 1)/(lamd_3 + lamd_3))];
            R       = P_i.' * R * P_i;
        end % end intsquaremat

        function R = intcarmat(self, tau, D, lamd)
            % following the Appendix Feunou and Okou 2016
            lamd_1  = D(1,1);       lamd_2  = D(2,2);       lamd_3  = D(3,3);
            R       = diag([exp(tau * lamd_1); exp(tau * lamd_2); exp(tau * lamd_3)]) * diag([self.intexp(tau, lamd - lamd_1); self.intexp(tau,lamd - lamd_2); self.intexp(tau,lamd - lamd_3)]);
        end % end intcarmat

        function R = intexp(self, tau, lamd)
            % following the Appendix Feunou and Okou 2016
            if lamd == 0;
                R   =   tau;
            else
                R   =   (exp(tau * lamd) - 1)/lamd;
            end;
        end % end intexp

        function R = int2carmat(self, tau, D, lamd)
            % following the Appendix Feunou and Okou 2016
            lamd_1  =   D(1,1);        lamd_2  =   D(2,2);         lamd_3  =   D(3,3);
            R       =   diag([exp(tau * lamd_1); exp(tau * lamd_2); exp(tau * lamd_3)]) * diag([self.int2exp(tau,lamd - lamd_1); self.int2exp(tau, lamd - lamd_2); self.int2exp(tau,lamd - lamd_3)]);
        end % end int2carmat

        function R = int2exp(self, tau, lamd)
            % following the Appendix Feunou and Okou 2016
            if lamd == 0;
                R   = 0.5 * tau^2;
            else
                R   = (tau * lamd * exp(tau * lamd) - exp(tau * lamd) + 1)/(lamd^2);
            end
        end % end int2exp

        function R = int2cubmat(self, tau, D, lamd, Gam)
            % following the Appendix Feunou and Okou 2016
            lamd_1  =  D(1,1);      lamd_2  =  D(2,2);      lamd_3 = D(3,3);
            R       =  diag([exp(tau * lamd_1); ...
                exp(tau * lamd_2); exp(tau * lamd_3)]) * [Gam(1,1) * self.int2exp(tau,lamd), Gam(1,2) * self.int2exp(tau,lamd + lamd_2 - lamd_1), Gam(1,3) * self.int2exp(tau,lamd + lamd_3 - lamd_1);...
                Gam(2,1) * self.int2exp(tau,lamd + lamd_1 - lamd_2), Gam(2,2) * self.int2exp(tau,lamd), Gam(2,3) * self.int2exp(tau,lamd+lamd_3-lamd_2);...
                Gam(3,1) * self.int2exp(tau,lamd + lamd_1 - lamd_3), Gam(3,2) * self.int2exp(tau,lamd + lamd_2 - lamd_3), Gam(3,3) * self.int2exp(tau,lamd)];
        end

        function R = intexplamd_beta_1(self, tau, lamd, A, C0)
            % following the Appendix Feunou and Okou 2016
            R   = (A + lamd * eye(3))\(expm(tau * (A + lamd * eye(3))) * C0 - C0) - (exp(tau * lamd) - 1) * (C0/lamd);
        end % end intexplamd_beta_1

        function Am_i = Am_i(self, A, lamd)
            % following the Appendix Feunou and Okou 2016
            Am          = A + lamd * eye(3);
            [P12,D12]   = eig(Am(1:2,1:2));     lamd_1 = D12(1,1);  lamd_2 = D12(2,2);
            lamd_3      = Am(3,3);              kappa_2=-lamd_3;
            P3          = [1/(lamd_1 + kappa_2), 0; 0, 1/(lamd_2 + kappa_2)] * P12 * [Am(3,1); Am(3,2)];
            P           = [P12, zeros(2,1); P3.', 1];
            P12_i       = (1/(P12(1,1) * P12(2,2) - P12(2,1) * P12(1,2))) * [P12(2,2), -P12(1,2); -P12(2,1), P12(1,1)];
            P_i         = [P12_i, zeros(2,1); -P3.' * P12_i, 0];
            D_i         = [1/lamd_1 0 0; 0 1/lamd_2 0; 0 0 1/lamd_3];
            Am_i        = P * D_i * P_i;
        end % end Am_i

    end

    methods

        %% Function AFT: constructs the AFT model
        %  Input:
        %    - varargin:        Variable input (see GARCH class)
        %  Output:
        %    - self:            AFT object
        %
        function self = AFT(varargin)
            self = self@GARCH();
            for no = 1:2:length(varargin)
                setfield(self, varargin{no}, varargin{no+1});
            end

            self.name      = 'AFT';
            self.cond_dist = 'Normal';

            % We remove the standard GARCH parameters to replace them with the
            % AFT model parameters
            self.removeParameter('sigma','alpha','beta','lambda');

            self.addParameter('eta',              0, [       0,      1]);
            self.addParameter('kappa1',           0, [       0,    100]);
            self.addParameter('nu1',              0, [       0,      1]);
            self.addParameter('sigma1',           0, [       0,      1]);
            self.addParameter('mu1',              0, [       0,    100]);
            self.addParameter('rho1',             0, [   -0.99,   0.99]);
            self.addParameter('v10',              0, [       0,      1]);

            self.addParameter('kappa2',           0, [       0,    100]);
            self.addParameter('nu2',              0, [       0,      1]);
            self.addParameter('sigma2',           0, [       0,      1]);
            self.addParameter('rho2',             0, [   -0.99,   0.99]);
            self.addParameter('v20',              0, [       0,      1]);

            self.addParameter('kappa3',           0, [       0,    100]);
            self.addParameter('mu3',              0, [       0,    100]);
            self.addParameter('rho3',             0, [       0,      1]);
            self.addParameter('v30',              0, [       0,      1]);

            self.addParameter('lambdap',           0, [       0,   500]);
            self.addParameter('lambdan',           0, [       0,   500]);

            self.addParameter('cp0',              0, [       0,     10]);
            self.addParameter('cp1',              0, [       0,   5000]);
            self.addParameter('cp2',              0, [       0,   5000]);
            self.addParameter('cp3',              0, [       0,   5000]);

            self.addParameter('cn0',              0, [       0,     10]);
            self.addParameter('cn1',              0, [       0,   5000]);
            self.addParameter('cn2',              0, [       0,   5000]);
            self.addParameter('cn3',              0, [       0,   5000]);

            self.addParameter('del2',             0, [       0,      2]);
            self.addParameter('del3',             0, [       0,      2]);
            self.addParameter('del4',             0, [       0,      2]);

            self.H_MAX = 1;
            % Update coefficients
            if ~isempty(self.moments)
                [Gamma0, Gamma1] = self.getCumulants();
                self.coeffs.Gamma0 = Gamma0;
                self.coeffs.Gamma1 = Gamma1;
            else
                warning('Be careful; cumulants not calculated.');
            end

        end % end AFT


        %% Function setParameterValues: set parameter values
        %  Input:
        %    - varargin:        Variable input (see GARCH class)
        %
        function [] = setParameterValues(self, varargin)
            setParameterValues@OptimProblem(self,varargin{:});

            % Update coefficients for moment calculation
            if ~isempty(self.moments)
                [Gamma0, Gamma1]     = self.getCumulants();
                self.coeffs.Gamma0   = Gamma0;
                self.coeffs.Gamma1   = Gamma1;
            else
                warning('Be careful; cumulants not calculated.');
            end
        end % end setParameterValues


        %% Function setPValues: set parameter values
        %  Input:
        %    - varargin:        Variable input (see GARCH class)
        %
        function [] = setPValues(self, varargin)
            setPValues@OptimProblem(self,varargin{:});
        end % end setPValues


        %% Function getPV: obtain parameter values (in a structure)
        %  Output:
        %    - pv:              Parameter structure
        %
        function pv = getPV(self)
            pv            = getPV@OptimProblem(self);
        end % end getPV


        %% Function unconditionalMean: calculates the unconditional mean and variance of the
        %  three factors as given by Feunou and Okou 2016, p.1018
        %  Input:
        %    - pv:              Current parameter value (structure)
        %  Output:
        %    - uncMean:             Unconditional mean of first, second, and third factors
        %    - uncVar:              Unconditional variance-covariance matrix
        function [uncMean, uncVar] = unconditionalMean(self, pv)
            h          = 1/252;
            lamd_b_m   = 2/(pv.lambdan^2);
            lamd_s_m   = 24/(pv.lambdan^4);


            % page 42 of the appendix
            Phi0       = h * [pv.kappa1 * pv.nu1 + pv.mu1 * lamd_b_m * pv.cn0; pv.kappa2 * pv.nu2; pv.mu3 * lamd_b_m * pv.cn0];
            K1         = h * [-pv.kappa1 + pv.mu1 * lamd_b_m * pv.cn1, pv.mu1 * lamd_b_m * pv.cn2, pv.mu1 * lamd_b_m * pv.cn3;...
                0, -pv.kappa2, 0; pv.mu3 * lamd_b_m * pv.cn1, pv.mu3 * lamd_b_m * pv.cn2,-pv.kappa3 + pv.mu3 * lamd_b_m * pv.cn3];
            Phi1       = eye(3) + K1;

            % obtain the inverse of K1
            detK1      = pv.kappa2 * (pv.kappa1 * pv.mu3 * lamd_b_m * pv.cn3 - pv.kappa3 * (pv.kappa1 - pv.mu1 * lamd_b_m * pv.cn1));
            K1_i       = (1/h) * (1/detK1) * [pv.kappa2 * (pv.kappa3 - pv.mu3 * lamd_b_m * pv.cn3), lamd_b_m * pv.mu1 * pv.cn2 * pv.kappa3, lamd_b_m * pv.mu1 * pv.cn3 * pv.kappa2;...
                0, pv.kappa3 * (pv.kappa1 - pv.mu1 * lamd_b_m * pv.cn1) - pv.kappa1 * pv.mu3 * lamd_b_m * pv.cn3, 0; lamd_b_m * pv.mu3 * pv.cn1 * pv.kappa2, lamd_b_m * pv.mu3 * pv.cn2 * pv.kappa1, pv.kappa2 * (pv.kappa1 - pv.mu1 * lamd_b_m * pv.cn1)];

            % Initial value for Vtt, and Ptt (page 42 of the appendix)
            uncMean    = - K1_i * Phi0;
            uncVar     = reshape((eye(9) - kron(Phi1,Phi1))\reshape(self.Gsig(uncMean, pv),[9,1]),3,3);

        end % end unconditionalMean


        %% Function Gsig: given vector V, the function calculates variance-covariance matrix of
        %  epsilon as given by Feunou and Okou 2016, p.1018
        %  Input:
        %    - V:               vector of unconditional variances (vector)
        %    - pv:              Current parameter value (structure)
        %  Output:
        %    - sig:             Unconditional mean of first, second, and third factors
        function [sig] = Gsig(self, V, pv)
            h          = 1/252;
            lamd_b_m   = 2/(pv.lambdan^2);
            lamd_s_m   = 24/(pv.lambdan^4);
            ct_n       = pv.cn0 + pv.cn1 * V(1) + pv.cn2 * V(2) + pv.cn3 * V(3);

            % Page 42 of appendix, (var(epsilon_{t+1}))
            sig        = h * [(pv.sigma1^2) * V(1) + (pv.mu1^2) * lamd_s_m * ct_n,0,pv.mu1 * pv.mu3 * (1 - pv.rho3) * lamd_s_m * ct_n;...
                0,(pv.sigma2^2) * V(2),0;...
                pv.mu1 * pv.mu3 * (1 - pv.rho3) * lamd_s_m * ct_n, 0, (pv.mu3^2) * ((1 - pv.rho3)^2 + pv.rho3^2) * lamd_s_m * ct_n];

        end % end Gsig


        %% Function getJumpIntensity: computes the jump intensity
        %  Input:
        %    - pv:              Current parameter value (structure)
        %    - v1:              First factor
        %    - v2:              Second factor
        %    - v3:              Third factor
        %  Output:
        %    - cp:              Positive jump intensity
        %    - cn:              Negative jump intensity
        function [cp, cn] = getJumpIntensity(self, pv, v1, v2, v3)
            cp = pv.cp0 + pv.cp1 .* v1 + pv.cp2 .* v2 + pv.cp3 .* v3;
            cn = pv.cn0 + pv.cn1 .* v1 + pv.cn2 .* v2 + pv.cn3 .* v3;
        end % end getJumpIntensity

        %% Function simulatePaths: generates sample path from the three-factor model
        %  Input:
        %    - nbPath:          Number of paths
        %    - T:               Length of each sample (in days)
        %    - S0:              Spot price
        %    - h1:              First factor initial value
        %    - h2:              Second factor initial value
        %    - h3:              Third factor initial value
        %  Output:
        %    - S:               Asset price
        %    - R:               Log-asset return
        %    - H1:              First variance factor
        %    - H2:              Second variance factor
        %    - H3:              Third variance factor
        %    - XP:              Positive jumps
        %    - XN:              Negative jumps
        %    - Y:               Independent jumps
        %
        function [S,R,H1,H2,H3,XP,XN,Y] = simulatePaths(self,nbPaths,T,S0,h1,h2,h3)
            % We get the right parameter structure (The model is under Q-measure!)
            pv = self.getPV();
            delta_t = 1/252;

            % We initialize vectors
            R  = NaN(nbPaths,T+1);
            H1 = NaN(nbPaths,T+1);
            H2 = NaN(nbPaths,T+1);
            H3 = NaN(nbPaths,T+1);
            XP = NaN(nbPaths,T);
            XN = NaN(nbPaths,T);
            Y  = NaN(nbPaths,T);

            % and we set their initial values
            R(:,1)  = 0;
            H1(:,1) = h1;
            H2(:,1) = h2;
            H3(:,1) = h3;

            % We compute the jump compensators
            xip     =  1/(pv.lambdap - 1);
            xin     = -1/(pv.lambdan + 1);

            % We compute the initial innovations and jump intensities
            [hp, hn]  = self.getJumpIntensity(pv, H1(:,1), H2(:,1), H3(:,1));
            innovations   = self.noiseSimulate(pv, nbPaths, H1(:,1), H2(:,1), H3(:,1), hp, hn);

            % For each day dt...
            for dt = 1:T
                XP(:,dt)        = innovations.xp;
                XN(:,dt)        = innovations.xn;
                Y(:,dt)         = innovations.y;

                Z1      = pv.rho1.*innovations.psi1 + sqrt(1-pv.rho1^2).*innovations.psi1perp;
                Z2      = pv.rho2.*innovations.psi2 + sqrt(1-pv.rho2^2).*innovations.psi2perp;
                Z3      = innovations.psi3;

                % R under risk-neutral measure!(This is not used in estimation!)
                R(:,dt+1)       = ((- 0.5) * H1(:,dt) ...
                    + (- 0.5) * H2(:,dt) ...
                    + (- 0.5* pv.eta^2) * H3(:,dt) ...
                    + (- xip) * hp ...
                    + (- xin) * hn )*delta_t ...
                    + Z1 + Z2 + pv.eta * Z3 + XP(:,dt) + XN(:,dt);

                H1(:,dt+1)      = boxed(H1(:,dt) + pv.kappa1*(pv.nu1 - H1(:,dt) )*delta_t + pv.sigma1*(innovations.psi1 ) + pv.mu1* XN(:,dt).^2, self.H_MIN, self.H_MAX);


                if h2 == 0
                    H2(:,dt+1)    = zeros(nbPaths,1);
                else
                    H2(:,dt+1)      = boxed(H2(:,dt) + pv.kappa2*(pv.nu2 - H2(:,dt) )*delta_t + pv.sigma2*(innovations.psi2 ) , self.H_MIN, self.H_MAX);
                end

                if h3 == 0
                    H3(:,dt+1)    = zeros(nbPaths,1);
                else
                    H3(:,dt+1) = boxed(H3(:,dt) + pv.kappa3.* (-H3(:,dt))*delta_t + pv.mu3*((1-pv.rho3).*XN(:,dt).^2+ pv.rho3.*(Y(:,dt).^2)), self.H_MIN, self.H_MAX);
                end

                [hp, hn]    = self.getJumpIntensity(pv, H1(:,dt+1), H2(:,dt+1), H3(:,dt+1));
                innovations     = self.noiseSimulate(pv, nbPaths, H1(:,dt+1), H2(:,dt+1), H3(:,dt+1), hp, hn);
            end
            S = S0.*exp(cumsum(R,2));
        end % end simulatePaths

        %% Function jumpSimulate : generates jumps
        %  Input:
        %    - pv:              Parameter values
        %    - nbPaths:         Number of paths
        %    - type:            .'n.' for negative, .'p.' for positive, .'y.' for
        %                       independent negative
        %    - intensity:       Jump arrival intensity
        %  Output:
        %    - jumps:           Jumps
        %    - N:               Number of jumps
        %
        function [jumps, N] = jumpSimulate(self, pv, nbPaths, type, intensity)
            % Draw the number of jumps from the Poisson distribution
            idx    = (1:nbPaths).';
            N      = poissrnd(intensity);

            switch type
                case 'n'
                    cumJumps    = - cumsum(exprnd(1/pv.lambdan,[nbPaths,max(N)]),2);
                case 'p'
                    cumJumps    =   cumsum(exprnd(1/pv.lambdap,[nbPaths,max(N)]),2);
            end

            % If there is no jumps or if there is at least a jump...
            if isempty(cumJumps)
                jumps             = zeros(nbPaths,1);
            else
                N1                = N;
                cumJumps(find(N1==0),:) = 0;
                N1(find(N1==0))   = 1;
                jumps             = cumJumps(sub2ind([nbPaths,max(N1)], idx, N1));
            end
        end % end jumpSimulate

        %% Function noiseSimulate: generates innovations for the three-factor model
        %  Input:
        %    - pv:              Current parameter value (structure)
        %    - nbPaths:         Number of paths
        %    - h1:              First variance factor
        %    - h2:              Second variance factor
        %    - h3:              Third variance factor
        %    - hp:              Positive jump intensity
        %    - hn:              Negative jump intensity
        %  Output:
        %    - innovations:     Structure containing innovations
        %
        function innovations = noiseSimulate(self, pv, nbPaths, h1, h2, h3, hp, hn)
            % Generate normal innovations \upsilon1=sqrt(h1)\epsilon1=\rho1 psi1 + sqrt(1-\rho1^2) psi1perp
            delta_t = 1/252;

            innovations.psi1         = randn([nbPaths,1]).* sqrt(delta_t*h1);
            innovations.psi1perp     = randn([nbPaths,1]).* sqrt(delta_t*h1);
            innovations.psi2         = randn([nbPaths,1]).* sqrt(delta_t*h2);
            innovations.psi2perp     = randn([nbPaths,1]).* sqrt(delta_t*h2);
            innovations.psi3         = randn([nbPaths,1]).* sqrt(delta_t*h3);

            % Define binary variables indicating the existance of jumps
            jump_p_exist = 1 - (((pv.cp0 == 0) && (pv.cp1 == 0) && (pv.cp2 == 0) && (pv.cp3 == 0)) || (pv.lambdap == 0));
            jump_n_exist = 1 - (((pv.cn0 == 0) && (pv.cn1 == 0) && (pv.cn2 == 0) && (pv.cn3 == 0)) || (pv.lambdan == 0));
            jump_y_exist = 1 - (((pv.cn0 == 0) && (pv.cn1 == 0) && (pv.cn2 == 0) && (pv.cn3 == 0)) || (pv.lambdan == 0));


            innovations.Np    = zeros(nbPaths,1);
            innovations.Nn    = zeros(nbPaths,1);
            innovations.Ny    = zeros(nbPaths,1);
            innovations.xp    = zeros(nbPaths,1);
            innovations.xn    = zeros(nbPaths,1);
            innovations.y     = zeros(nbPaths,1);

            if (jump_y_exist ==1)
                [innovations.y, innovations.Ny]   = self.jumpSimulate(pv, nbPaths, 'n', hn*delta_t);
            end
            if (jump_n_exist ==1)
                [innovations.xn, innovations.Nn]  = self.jumpSimulate(pv, nbPaths, 'n', hn*delta_t);
            end
            if (jump_p_exist ==1)
                [innovations.xp, innovations.Np]  = self.jumpSimulate(pv, nbPaths, 'p', hp*delta_t);
            end
        end % end noiseSimulate


        %% Function logLikelihood: computes the likelihood function
        %  Input:
        %    - pvalues:         Current parameter value
        %    - method:          Method (object) for the optimization (kfdnf or fokf)
        %  Output:
        %    - loglike:         Log-likelihood value
        %    - filtered:        Filtered variables
        %
        function [loglike, filtered] = logLikelihood(self, pvalues, method)
            self.setPValues(pvalues);
            [loglike, filtered] = method.logLikelihood();
        end % end logLikelihood


        %% Function maximumLikelihood: maximizes the likelihood function
        %  Input:
        %    - optmethod:       Optimization method (fmincon, fminsearch,
        %                       patternsearch)
        %    - method:          Method for the optimization (kfdnf or fokf)
        %  Output:
        %    - results:         Optimization results
        %
        function results = maximumLikelihood(self, optmethod, method)
            if strcmp(optmethod,'fmincon')
                self.mle = self.fmincon(method);
            elseif strcmp(optmethod,'fminsearch')
                self.mle = self.fminsearch(100,method);
            elseif strcmp(optmethod,'patternsearch')
                self.mle = self.patternsearch(method);
            else
                error('Method not implemented yet.');
            end
            results = self.mle;
        end % end maximumLikelihood


        %% Function objective : computes objective function value
        %  Input:
        %    - x:               Value of the parameters
        %    - varargin:        Varible input argument structure
        %  Output:
        %    - nll:             Negative log-likelihood value
        %
        function nll = objective(self, x, varargin)
            loglikelihood   = self.logLikelihood(x, varargin{:});
            nll             = -sum(loglikelihood);
        end % end objective

        %% Function getCumulants : computes risk-neutral cumulants
        %  Input:
        %  Output: cum = Gamma0 + Gamma1 * h
        %    - Gamma0:              Constant coefficient to get cumulants
        %    - Gamma1:              Coefficient
        function [Gamma0, Gamma1] = getCumulants(self)
            % Calculate alphas and betas
            [alpha2,beta2,alpha3,beta3,alpha4,beta4]  = self.AFT_VSK_Coeff();
            % Get Gamma0 and Gamma1
            Gamma0        = [alpha2.';alpha3.';alpha4.'];
            Gamma1        = [beta2(1,:).',  beta2(3,:).', beta2(2,:).';beta3(1,:).',  beta3(3,:).', beta3(2,:).';beta4(1,:).',  beta4(3,:).', beta4(2,:).'];
        end % end getCumulants

        %% Function getMoments : computes risk-neutral moments
        %  Input:
        %    - h1i:             The risk-neutral value of the first factor
        %    - h2i:             The risk-neutral value of the second factor
        %    - h3i:             The risk-neutral value of the third factor
        %    - force:           Boolean that forces the recalculation of the
        %                       coefficients
        %  Output:
        %    - Mivol:         Model implied variance values
        %    - Miskew:        Model implied skewness values
        %    - Mikurt:        Model implied kurtosis values
        %
        function [Mivol,Miskew,Mikurt] = getMoments(self,h1i,h2i,h3i,force)
            if nargin < 5; force = true; end

            h1i = h1i(:);
            h2i = h2i(:);
            h3i = h3i(:);

            numMat = numel(self.moments.Maturity);

            if force || isempty(self.coeffs)
                [Gamma0,Gamma1] = self.getCumulants();
                self.coeffs.Gamma0 = Gamma0;
                self.coeffs.Gamma1 = Gamma1;
            else
                Gamma0 = self.coeffs.Gamma0;
                Gamma1 = self.coeffs.Gamma1;
            end

            cummom     = Gamma0 + Gamma1 * [h1i.';h2i.';h3i.'];
            Cum2_t     = cummom(1:numMat,:);
            Cum3_t     = cummom(numMat+1:2*numMat,:);
            Cum4_t     = cummom(2*numMat+1:3*numMat,:);
            Mivol      = (sqrt(Cum2_t)).';
            Miskew     = (Cum3_t./(Mivol.'.^3)).';
            Mikurt     = (3+Cum4_t./(Mivol.'.^4)).';
        end % end getMoments


        %% Function getMoments : computes risk-neutral alphas and betas used in Gamma0 and Gamma1
        %  Input:
        %  Output:
        %    - alpha_2:       Second derivative of alpha with respect to u
        %    - beta_2:        Second derivative of beta with respect to u
        %    - alpha_3:       Third derivative of alpha with respect to u
        %    - beta_3:        Third derivative of beta with respect to u
        %    - alpha_4:       Fourth derivative of alpha with respect to u
        %    - beta_4:        Fourth derivative of beta with respect to u
        function [alpha_2,beta_2,alpha_3,beta_3,alpha_4,beta_4] = AFT_VSK_Coeff(self)
            tau         = self.moments.Maturity;
            alpha_2     = zeros(1,numel(tau));alpha_3=zeros(1,numel(tau));alpha_4=zeros(1,numel(tau));
            beta_2      = zeros(3,numel(tau));beta_3=zeros(3,numel(tau));beta_4=zeros(3,numel(tau));

            for j=1:numel(tau)
                [alpha_2(j),beta_2(:,j),alpha_3(j),beta_3(:,j),~,~] = self.AFT_RNM_Coeff(tau(j));
                f            = self.AFT_KURT_Coeff(tau(j));
                alpha_4(j)   =f(1);beta_4(:,j)=f(2:end);


                alpha_2(j)  = alpha_2(j);    alpha_3(j)  = alpha_3(j);    alpha_4(j)  = alpha_4(j);
                beta_2(:,j) = beta_2(:,j);   beta_3(:,j) = beta_3(:,j);  beta_4(:,j)  = beta_4(:,j);
            end
        end % end AFT_VSK_Coeff



        function initial  = getInitial(self, seed)
            rng(seed)
            [pvalues,bounds,~] = self.getPValues();
            u = unifrnd(0,1,length(pvalues),1);

            u = bounds(:,1) + (bounds(:,2) - bounds(:,1)).*u;
            self.setPValues(u)
            pv        = self.getPV();
            [Gam0, Gam1] = self.getCumulants();
            if (pv.kappa2 <= 0) || (pv.kappa1 <= (2*pv.mu1*pv.cn1)/pv.lambdan^2) || (pv.kappa3 <= (2*pv.mu3*pv.cn3*pv.kappa1)/(pv.kappa1*pv.lambdan^2 - 2*pv.mu1*pv.cn1)) || (pv.sigma1^2 > 2*pv.kappa1*pv.nu1) || (pv.sigma2^2 > 2*pv.kappa2*pv.nu2) || (sum(isnan(Gam0))~=0) || (sum(sum(isnan(Gam1)))~=0)      initial  = [];
                self.setPValues(zeros(length(pvalues),1))
            else
                fieldNames  = fieldnames(pv);
                fieldValues = struct2cell(pv);
                initial     = reshape([fieldNames, fieldValues].', 1, []);
            end
        end



        %% Function NumJacobian: compute the symmetric numerical first order derivatives of a
        % multivariate function.
        %
        % Inputs:
        %     - x:          A point (d x 1) at which the derivatives will be computed;
        %     - prec:       Percentage of +\- around x (in fraction).
        %
        % Output:
        %     - J:          derivatives (N x d)
        %
        function J = NumJacobian( self, x, prec, method )

            pv     = self.getPV();
            fnames = fieldnames(self.params);
            d      = length(fnames)
            count = 0;

            J = zeros(d,length(self.series));
            fixed_ind = ones(d,1);

            for ii = 1:d
                ii
                if self.params.(fnames{ii}).fixed == 0
                    fixed_ind(ii) = 0;

                    orig =  self.params.(fnames{ii}).value;
                    orig = {fnames{ii}, orig};
                    x2   = x.(fnames{ii});
                    x1   = x.(fnames{ii});

                    x1 = x1 - prec* x1;
                    x2 = x2;

                    fn1    = {fnames{ii}, x1};
                    self.setParameterValues(fn1{:});
                    pv = self.getPV();
                    cc = 0;
                    while (pv.kappa2 <= 0) || (pv.kappa1 <= (2*pv.mu1*pv.cn1)/pv.lambdan^2) || (pv.kappa3 <= (2*pv.mu3*pv.cn3*pv.kappa1)/(pv.kappa1*pv.lambdan^2 - 2*pv.mu1*pv.cn1)) || (pv.sigma1^2 > 2*pv.kappa1*pv.nu1) || (pv.sigma2^2 > 2*pv.kappa2*pv.nu2)
                        cc = cc+1;
                        x1     = x.(fnames{ii});
                        x1     = x1 - prec*0.1^cc* x1;
                        fn1    = {fnames{ii}, x1};
                        self.setParameterValues(fn1{:});
                        pv = self.getPV();
                    end

                    [pvalues,bounds] = self.getPValues();
                    ll1  = self.logLikelihood(pvalues, method);

                    fn2    = {fnames{ii}, x2};
                    self.setParameterValues(fn2{:});

                    [pvalues,bounds] = self.getPValues();
                    ll2  = self.logLikelihood(pvalues, method);

                    J(ii ,:) = ( ll1 - ll2 ) ./ ( x1 - x2 );

                end

            end

            J(find(fixed_ind) ,:) = [];

        end % end NumJacobian

    end % methods

end % AFT



