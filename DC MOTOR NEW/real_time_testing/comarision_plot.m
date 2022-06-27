PI=load('PI_1500_step_with_error.mat');
quad_1= load('quad_agnet1_1500_step.mat');
quad_2= load('quad_agnet2_1500_step.mat');

figure(1);
plot(PI.out.Speed,'LineWidth',2);
hold on;
plot(PI.out.Setpoint,'LineWidth',2);
plot(quad_1.out.rpm_real,'LineWidth',2);
plot(quad_2.out.rpm_real,'LineWidth',2);
xlabel('Time(sec)');
ylabel('RPM');
title('Comparision of controllers RPM')
legend('PI controller','Setpoint','DDPG quadratic agent-1','DDPG quadratic agent-2');

figure(2);
plot(PI.out.PWM,'LineWidth',2);
hold on;
plot(quad_1.out.pwm_real,'LineWidth',2);
plot(quad_2.out.pwm_real,'LineWidth',2);
xlabel('Time(sec)');
ylabel('PWM');
title('Comparision of controllers PWM')

legend('PI controller','DDPG quadratic agent-1','DDPG quadratic agent-2');

figure(3);
plot(PI.out.error,'LineWidth',2);
hold on;

plot(quad_1.out.error_real,'LineWidth',2);
plot(quad_2.out.error_real,'LineWidth',2);
xlabel('Time(sec)');
ylabel('Error');
title('Comparision of controllers Error')

legend('PI controller','DDPG quadratic agent-1','DDPG quadratic agent-2');

