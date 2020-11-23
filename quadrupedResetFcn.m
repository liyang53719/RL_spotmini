% Helper function to reset walking robot simulation with different initial conditions
%
% Copyright 2019 The MathWorks, Inc.

function in = quadrupedResetFcn(in)
% Randomization Parameters

    l1 = evalin('base','l1');
    l2 = evalin('base','l2');

    max_foot_disp_x = 0.1;
    min_body_height = 0.7;
    max_body_height = 0.8;
    max_speed_x = 0.05;
    max_speed_y = 0.025;
    
%     y_body = 0;
%     vx = 0;
%     vy = 0;
%     th_FL = zeros(1,2);
%     th_FR = zeros(1,2);
%     th_RL = zeros(1,2);
%     th_RR = zeros(1,2);
    
    % Chance of randomizing initial conditions
    if rand < 0.5
        % Randomize height
        b = min_body_height + (max_body_height - min_body_height) * rand;
        
        % Randomize x-displacement of foot from hip joint        
        a = -max_foot_disp_x + 2 * max_foot_disp_x * rand(1,4);
        
        % Calculate joint angles
        d2r = pi/180;
        th_FL = d2r * quadrupedInverseKinematics(a(1),-b,l1,l2);
        th_FR = d2r * quadrupedInverseKinematics(a(2),-b,l1,l2);
        th_RL = d2r * quadrupedInverseKinematics(a(3),-b,l1,l2);
        th_RR = d2r * quadrupedInverseKinematics(a(4),-b,l1,l2);
        
        % Adjust for foot height
        foot_height = 0.05*l2*(1-sin(2*pi-(3*pi/2+sum([th_FL;th_FR;th_RL;th_RR],2))));
        %s2ph = 0.05-l2*0.1/2*(1-sind(90-(-th_FL(2)+pi)*180/pi-(-th_FL(1))*180/pi));
        y_body = max(b) + max(foot_height);
        
        % Randomize body velocities
        vx = 2 * max_speed_x * (rand-0.5);
        vy = 2 * max_speed_y * (rand-0.5); 
        
    % Chance of starting from default initial conditions
    else        
        y_body = 0.7588;
        th_FL = [-0.8234 1.6468];
        th_FR = th_FL;
        th_RL = th_FL;
        th_RR = th_FL;
        vx = 0;
        vy = 0;
    end
    
    in = setVariable(in,'y_init',y_body);
    in = setVariable(in,'init_ang_FL',th_FL);
    in = setVariable(in,'init_ang_FR',th_FR);
    in = setVariable(in,'init_ang_RL',th_RL);
    in = setVariable(in,'init_ang_RR',th_RR);
    in = setVariable(in,'vx_init',vx);
    in = setVariable(in,'vy_init',vy);
    
end