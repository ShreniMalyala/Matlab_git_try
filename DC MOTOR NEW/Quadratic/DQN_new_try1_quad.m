mdl='ddpg_new_try1_sim_quad';
agentblk='ddpg_new_try1_sim_quad/RL Agent';
obsInfo = rlNumericSpec([3 1],...
    'LowerLimit',[-inf -1000 0 ]',...
    'UpperLimit',[ inf 5000 3500]');
obsInfo.Name = 'observations';
obsInfo.Description = 'integrated error, error, RPM';
numObservations= obsInfo.Dimension(1);

actInfo = rlFiniteSetSpec([1 1]);
actInfo.Name = 'PWM';
numActions=actInfo.Dimension(1);

env= rlSimulinkEnv(mdl,agentblk,obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn_ref(in);

%% 

dnn = [
    featureInputLayer(obsInfo.Dimension(1),'Normalization','none','Name','state')
    fullyConnectedLayer(50,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(25, 'Name','CriticStateFC2')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(length(actInfo.Elements),'Name','output')];
criticOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',1);
critic = rlQValueRepresentation(dnn,obsInfo,actInfo,'Observation',{'state'},criticOpts);
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false, ...    
    'TargetSmoothFactor',1, ...
    'TargetUpdateFrequency',4, ...   
    'ExperienceBufferLength',100000, ...
    'DiscountFactor',0.99, ...
    'MiniBatchSize',256);
agent = rlDQNAgent(critic,agentOpts);
%% 
Ts = 0.01;
Tf = 2;
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',1000, ...
    'MaxStepsPerEpisode',500, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',480); 
%% 

doTraining = true;
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
%else
    % Load the pretrained agent for the example.
    
end
