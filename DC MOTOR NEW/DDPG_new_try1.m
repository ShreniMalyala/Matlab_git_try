
mdl='ddpg_new_try1_sim';
agentblk='ddpg_new_try1_sim/RL Agent';

actInfo = rlNumericSpec([1 1], 'LowerLimit',[0]', 'UpperLimit',[255]');
actInfo.Name = 'PWM';
numActions=actInfo.Dimension(1);

obsInfo = rlNumericSpec([3 1],...
    'LowerLimit',[-inf -1000 0 ]',...
    'UpperLimit',[ inf 5000 3500]');
obsInfo.Name = 'observations';
obsInfo.Description = 'integrated error, error, RPM';
numObservations= obsInfo.Dimension(1);

env= rlSimulinkEnv(mdl,agentblk,obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn_ref(in);

%% 
Ts = 0.1;
%Ts =0.015;
%Tf =7.5;
Tf = 60;
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(50,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(25,'Name','CriticStateFC2')];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','Action')
    fullyConnectedLayer(25,'Name','CriticActionFC1')];
    
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
figure(1)
plot(criticNetwork)

criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);

actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(25, 'Name','actorFC1')
    reluLayer('Name','actorRelu')
    fullyConnectedLayer(25, 'Name','actorFC2')
    tanhLayer('Name','actorTanh')
    fullyConnectedLayer(numActions,'Name','Action')
    ];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);

figure(2)
plot(layerGraph(actorNetwork))

agentOpts = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',1.0, ...
    'MiniBatchSize',64, ...
    'ExperienceBufferLength',1e6); 
agentOpts.NoiseOptions.Variance = 0.2;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;
 
agent = rlDDPGAgent(actor,critic,agentOpts);
%% 

% Training:

maxepisodes = 500;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',50, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',30000);
%% 

doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
%else
    % Load the pretrained agent for the example.
    
end



