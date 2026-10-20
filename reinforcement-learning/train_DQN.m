clc; clear; close all;

%% 환경 설정
grid_size = 100; % 100x100 격자
num_actions = 9; % 8방향 이동 + 정지
num_episodes = 500; % 학습 에피소드 수
max_steps = 100; % 한 에피소드당 최대 스텝 수

% 강화학습 환경 생성
env = FireSpreadEnv(grid_size, num_actions);

%% 기상 데이터 및 입목도 데이터 로드
data = readtable('generated_weather_data.csv');
weather_matrix = reshape(table2array(data(:,3:6)), grid_size, grid_size, [], 4); % 온도, 풍속, 풍향, 습도

% 입목도 데이터 로드
tree_data = readtable('tree_density_data.csv');
tree_density_matrix = zeros(grid_size, grid_size);

% row_index, col_index를 기반으로 입목도 행렬 생성
for i = 1:height(tree_data)
    r = tree_data.row_index(i) + 1;
    c = tree_data.col_index(i) + 1;
    tree_density_matrix(r, c) = tree_data.mean(i);
end

%% DQN 에이전트 설정
% 환경 정보 가져오기
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

% 신경망 구성
layers = [
    featureInputLayer(prod(obsInfo.Dimension), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(32, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(32, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
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

% 환경에 기상 데이터 및 입목도 데이터 반영
for t = 1:size(weather_matrix,3)
    env.FuelMap = weather_matrix(:,:,t,1) .* tree_density_matrix; % 온도와 입목도를 연료로 반영
    trainStats = train(agent, env, trainingOpts);
end

%% 학습된 에이전트 테스트
num_test_episodes = 10;
for i = 1:num_test_episodes
    sim(env, agent);
end

display("Training completed!");