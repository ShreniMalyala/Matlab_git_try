mdl='A2C_new_2ob_try1_sim_quad';
agentblk='A2C_new_2ob_try1_sim_quad/RL Agent';

obsInfo = rlNumericSpec([2 1],...
    'LowerLimit',[-1000 0 ]',...
    'UpperLimit',[5000 3500]');
obsInfo.Name = 'observations';
obsInfo.Description = 'error, RPM';
numObservations= obsInfo.Dimension(1);

actInfo = rlNumericSpec([1 1],'LowerLimit',[0]', 'UpperLimit',[255]' );
actInfo.Name = 'PWM';
numActions=actInfo.Dimension(1);

env= rlSimulinkEnv(mdl,agentblk,obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn_ref_A2C(in);
%%
initOpts = rlAgentInitializationOptions('NumHiddenUnit',128);
agent = rlSACAgent(obsInfo,actInfo,initOpts);
actorNet = getModel(getActor(agent));
criticNet = getModel(getCritic(agent));

figure(1)
plot(actorNet)
figure(2)
plot(criticNet)

%%
Ts=0.01;
Tf=20;
maxepisodes = 500;
maxsteps= ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',10, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',2000);
%%
trainingStats = train(agent,env,trainOpts);

