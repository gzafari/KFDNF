classdef Param < handle
% Handle class that serves as a wrapper for parameters being optimized in a OptimProblem  
  properties 
    name   = '';    
    value  = NaN;
    fixed  = false;
    bounds = [-Inf +Inf];
    
    scale_slope = 2;     % Sigmoid function
    inverse_sig_max = 5; % Optim domain will be in [-ISM, +ISM], ISM = inverse_sig_max
  end % end properties
    
  methods 
    function self = Param(name, value, bounds)      
      self.name = name;      
      if nargin > 1
        self.value = value;
        if nargin > 2
          self.bounds = bounds;
        end
      end
    end    
    
    function scaled = getScaledValue(self)
      a = self.bounds(1);
      b = self.bounds(2);
      ss = self.scale_slope;
      val = self.value;
      
      % inverse sigmoid - Domain is real axis
      scaled =  log((val-a) ./ (b-val)) ./ ss;
    end        
    
    function value = setScaledValue(self, scaled)      
      a = self.bounds(1);
      b = self.bounds(2);      
      ss = self.scale_slope;
      value = a  +  (b - a) ./ (1 + exp(-ss.*scaled));
      self.value = value;
    end
    
    function [x,y,x1,y1] = illustrateScalingSpecific(self)
      x = [0.00001, 0.0005, 0.01:0.001:0.99, 0.9995, 0.99999]'; 
      x = self.bounds(1) + x.*diff(self.bounds);
      y = self.scale(x, self.bounds(1), self.bounds(2));
      z = self.unscale(y, self.bounds(1), self.bounds(2));
      x1 = self.value;
      y1 = self.getScaledValue();
%       
%       figure
%       subplot(3,1,1);
%       plot(x, y)
%       hold on
%       plot(self.value, self.getScaledValue(),'r.','MarkerSize',12);
%       hold off
%       box on
%       xlabel('Value'); ylabel('Scaled');      
%     
%       subplot(3,1,2); hold on;
%       plot(x, [x, z]); 
%       legend('45-degree line', 'Recovered', 'Location','SE');
%       xlabel('Value'); ylabel('Recovered value');      
% 
%       subplot(3,1,3);
%       plot(x, z-x)
%       xlabel('Value'); ylabel('Error on value');      
      
    end
    

    
  end % methods

  methods (Static)   
    function [scaled] = scale(x, lb, ub)
      obj = Param('x', x, [lb, ub]);
      scaled = obj.getScaledValue();
    end
    
    function [value] = unscale(y, lb, ub)
      obj = Param('x', NaN, [lb, ub]);
      value = obj.setScaledValue(y);
    end    
    
    function [] = illustrateScaling()
      x = [0.00001, 0.0005, 0.01:0.001:0.99, 0.9995, 0.99999]'; 
      y = Param.scale(x, 0,1);
      z = Param.unscale(y, 0,1);
      
      figure
      subplot(3,1,1);
      plot(x, y)
      xlabel('Value'); ylabel('Scaled');      
    
      subplot(3,1,2); hold on;
      plot(x, [x, z]); 
      legend('45-degree line', 'Recovered', 'Location','SE');
      xlabel('Value'); ylabel('Recovered value');      

      subplot(3,1,3);
      plot(x, z-x)
      xlabel('Value'); ylabel('Error on value');      
      
      figure
      ISM = Param('default').inverse_sig_max;
      y = -ISM:0.1:ISM;
      for no = 1:4
        subplot(4,1,no);
        Mx = 10^(no-1);      
        z = Param.unscale(y, -Mx,Mx);
        plot(y, z)
        xlabel('Optim value'); 
        ylabel({'Param value'; sprintf('[-%d, %d]',Mx,Mx)});
        DIFF = [z(1)+Mx, z(end)-Mx] ./ Mx
      end
    
      figure
      y = -5:0.1:5;
      for no = 1:4
        subplot(4,1,no);
        Mx = 10^(no-1);      
        z = Param.unscale(y, 0,Mx);
        plot(y, z)
        xlabel('Optim value'); 
        ylabel({'Param value'; sprintf('[0, %d]',Mx)});
        DIFF = [z(1), z(end)-Mx] ./ Mx
      end      
    end
    
  end % methods (Static)
end
