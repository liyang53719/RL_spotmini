function [angles,A,B,C,ang1,ang2] = quadrupedInverseKinematics(a,b,L1,L2)
% Performs inverse kinematics for one limb, restricts solution to
% knee-backward configuraiton
%   l1 and l2 are the upper and lower limb lengths
%   a is the x-displacement of the foot from the hip joint
%   b is the y-displacement of the foot from the hip joint
%
% Assume origin at hip joint.
%
% Output angles in degrees.
%
    % Apply bounds on a. For a given height b, a is bounded
    if abs(b) < L1
        max_a = L2 - L1*sind(acosd(abs(b)/L1));   % knee at ground level
        min_a = -sqrt(L1^2+L2^2-b^2);           % feet off-ground
        if a > max_a
            a = max_a;
        end
        if a < min_a
            a = min_a;
        end
    end
    
    % Check for singularity, if (a,b) is out of the workspace
    if sqrt(a^2+b^2) > (L1+L2)
        disp('Singularity');
        angles = [Inf Inf];
    else
        d2r = pi/180;
        phii = atan2(b,a)/d2r;
        if phii < 0
            phii = 360 + phii;
        end
        B = acosd((L1^2+L2^2-a^2-b^2)/(2*L1*L2));
        A = acosd((L1^2-L2^2+a^2+b^2)/(2*L1*sqrt(a^2+b^2)));
        C = acosd((L2^2-L1^2+a^2+b^2)/(2*L2*sqrt(a^2+b^2)));
        ang1 = phii - A;
        ang2 = phii + C;
        hip_ang = ang1 - 270;
        knee_ang = ang2 - ang1; %180 + A + C;
        angles = [hip_ang knee_ang];
    end
    
end


