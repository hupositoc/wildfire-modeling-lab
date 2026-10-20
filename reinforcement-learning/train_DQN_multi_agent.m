clc; clear; close all;

%% 환경 설정
grid_size = 100; % 100x100 격자
num_agents = 5; % 멀티 에이전트 수
num_actions = 9; % 8방향 이동 + 정지
num_episodes = 500; % 학습 에피소드 수
max_steps = 200; % 한 에피소드당 최대 스텝 수

% 멀티에이전트 환경 생성
env = MultiAgentFireEnv(grid_size, num_agents);

%% DQN 에이전트 설정
% 환경 정보 가져오기
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% 신경망 구성
layers = [
    featureInputLayer(obsInfo.Dimension(2), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(24, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(numel(actInfo.Elements), 'Name', 'output')
];

% 옵션 설정
options = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);

% 신경망 기반 Q-표현 생성
critic = rlQValueRepresentation(layers, obsInfo, actInfo, ...
    'Observation',{'state'}, 'Action',{'output'}, options);

% DQN 에이전트 설정
explorationOptions = rl.option.EpsilonGreedyExploration(...
    'Epsilon', 1.0, 'EpsilonDecay', 0.995, 'EpsilonMin', 0.05);

agentOpts = rlDQNAgentOptions(...
    "SampleTime", 1, ...
    "DiscountFactor", 0.99, ...
    "EpsilonGreedyExploration", explorationOptions);
agent = rlDQNAgent(critic, agentOpts);

% 학습 옵션 설정 및 에이전트 학습
trainingOpts = rlTrainingOptions(...
    "MaxEpisodes", num_episodes, ...
    "MaxStepsPerEpisode", max_steps, ...
    "Verbose", true, ...
    "StopTrainingCriteria", "AverageReward", ...
    "StopTrainingValue", 100);
trainStats = train(agent, env, trainingOpts);

%% 학습된 에이전트 테스트
num_test_episodes = 10;
for i = 1:num_test_episodes
    sim(env, agent);
end

display("Training completed!");
