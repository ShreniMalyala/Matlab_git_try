mdl='A2C_new_3ob_try1_sim_quad';
agentblk='A2C_new_3ob_try1_sim_quad/RL Agent';

obsInfo = rlNumericSpec([3 1],...
    'LowerLimit',[-inf -1000 0 ]',...
    'UpperLimit',[inf 5000 3500]');
obsInfo.Name = 'observations';
obsInfo.Description = 'Integrated_error, error, RPM';
numObservations= obsInfo.Dimension(1);

actInfo = rlFiniteSetSpec([1 1] );
actInfo.Name = 'PWM';
numActions=actInfo.Dimension(1);

env= rlSimulinkEnv(mdl,agentblk,obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn_ref_A2C(in);
%%
statePath = [
    featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(24, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(24, 'Name', 'CriticStateFC2')];
actionPath = [
    featureInputLayer(actInfo.Dimension(1), 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(24, 'Name', 'CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1, 'Name', 'output')];
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');

% set some options for the critic
criticOpts = rlRepresentationOptions('LearnRate',0.01,'GradientThreshold',1);

% create the critic based on the network approximator
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'action'},criticOpts);
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN',false, ...    
    'TargetUpdateMethod',"periodic", ...
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
