mdl='A2C_new_2ob_try1_sim_quad';
agentblk='A2C_new_2ob_try1_sim_quad/RL Agent';

obsInfo = rlNumericSpec([2 1],...
    'LowerLimit',[-1000 0 ]',...
    'UpperLimit',[5000 3500]');
obsInfo.Name = 'observations';
obsInfo.Description = 'Integrated error, error, RPM';
numObservations= obsInfo.Dimension(1);

actInfo = rlNumericSpec([1 1],'LowerLimit',[0]', 'UpperLimit',[255]' );
actInfo.Name = 'PWM';
numActions=actInfo.Dimension(1);

env= rlSimulinkEnv(mdl,agentblk,obsInfo,actInfo);
env.ResetFcn = @(in)localResetFcn_ref_A2C(in);
%%
Ts = 0.01;
Tf = 10;

cnet = [
    featureInputLayer(numObservations,"Normalization","none","Name","observation")
    fullyConnectedLayer(128,"Name","fc1")
    concatenationLayer(1,2,"Name","concat")
    reluLayer("Name","relu1")
    fullyConnectedLayer(64,"Name","fc3")
    reluLayer("Name","relu2")
    fullyConnectedLayer(32,"Name","fc4")
    reluLayer("Name","relu3")
    fullyConnectedLayer(1,"Name","CriticOutput")];
actionPath = [
    featureInputLayer(numActions,"Normalization","none","Name","action")
    fullyConnectedLayer(128,"Name","fc2")];

% Connect the layers.
criticNetwork = layerGraph();
criticNetwork= addLayers(criticNetwork, cnet);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = connectLayers(criticNetwork,"fc2","concat/in2");

figure(1)
plot(criticNetwork);

criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
%criticdlnet = dlnetwork(criticNetwork);
%critic = rlQValueFunction(criticNetwork,obsInfo,actInfo, ...
%    "ObservationInputNames","observation");
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'});

% Create the actor network layers.
anet = [
    featureInputLayer(numObservations,"Normalization","none","Name","observation")
    fullyConnectedLayer(128,"Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(64,"Name","fc2")
    reluLayer("Name","relu2")];
meanPath = [
    fullyConnectedLayer(32,"Name","meanFC")
    reluLayer("Name","relu3")
    fullyConnectedLayer(numActions,"Name","mean")];
stdPath = [
    fullyConnectedLayer(numActions,"Name","stdFC")
    reluLayer("Name","relu4")
    softplusLayer("Name","std")];

% Connect the layers.
actorNetwork = layerGraph(anet);
actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,stdPath);
actorNetwork = connectLayers(actorNetwork,"relu2","meanFC/in");
actorNetwork = connectLayers(actorNetwork,"relu2","stdFC/in");

figure(2)
plot(actorNetwork)

%actordlnet = dlnetwork(actorNetwork);

actor = rlStochasticActorRepresentation(actorNetwork, obsInfo, actInfo, ...
    "ObservationInputNames","observation");

agentOpts = rlSACAgentOptions( ...
    "SampleTime",Ts, ...
    "TargetSmoothFactor",1e-3, ...    
    "ExperienceBufferLength",1e6, ...
    "MiniBatchSize",128, ...
    "NumWarmStartSteps",1000, ...
    "DiscountFactor",0.99);
agentOpts.ActorOptimizerOptions.Algorithm = "adam";
agentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
agentOpts.ActorOptimizerOptions.GradientThreshold = 1;

agentOpts.CriticOptimizerOptions.Algorithm = "adam";
agentOpts.CriticOptimizerOptions.LearnRate = 1e-4;
agentOpts.CriticOptimizerOptions.GradientThreshold = 1;

agent = rlSACAgent(actor,critic,agentOpts);

trainOpts = rlTrainingOptions(...
    "MaxEpisodes", 5000, ...
    "MaxStepsPerEpisode", floor(Tf/Ts), ...
    "ScoreAveragingWindowLength", 100, ...
    "Plots", "training-progress", ...
    "StopTrainingCriteria", "AverageReward", ...
    "StopTrainingValue", 675);
%%
doTraining = true;
if doTraining
    stats = train(agent,env,trainOpts);
else
    %load("kinovaBallBalanceAgent.mat")       
end