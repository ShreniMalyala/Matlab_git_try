mdl='ddpg_new_try1_sim_quad';
agentblk='ddpg_new_try1_sim_quad/RL Agent';

obsInfo = rlNumericSpec([3 1],...
    'LowerLimit',[-inf -1000 0 ]',...
    'UpperLimit',[ inf 5000 3500]');
obsInfo.Name = 'observations';
obsInfo.Description = 'Integrated error, error, RPM';
numObservations= obsInfo.Dimension(1);

actInfo = rlNumericSpec([1 1],'LowerLimit',[0]', 'UpperLimit',[255]' );
actInfo.Name = 'PWM';
numActions=actInfo.Dimension(1);

env= rlSimulinkEnv(mdl,agentblk,obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn_ref(in);
%%

criticNet=[featureInputLayer(numObservations,'Normalization','none','Name','state')
    fullyConnectedLayer(25,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(25,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(25,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','out')];
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);

critic = rlValueRepresentation(criticNet,obsInfo,'Observation',{'state'},criticOpts);

actorNet= [featureInputLayer(numObservations,'Normalization','none','Name','state')
    fullyConnectedLayer(150,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(2,'Name','out')];
actorOpts = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);

actor = rlStochasticActorRepresentation(actorNet,obsInfo,actInfo,...
    'Observation',{'state'},actorOpts);

agentOpts = rlACAgentOptions('NumStepsToLookAhead',32,'DiscountFactor',0.99);
agent = rlACAgent(actor,critic,agentOpts);

%% 
% Training:
Ts=0.1;
Tf=2;
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