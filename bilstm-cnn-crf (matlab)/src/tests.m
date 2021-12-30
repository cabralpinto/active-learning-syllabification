%% configuration
clear;
config = jsondecode(fileread("config.json"));
%% data
% load dataset
text = insertAfter(fileread(fullfile("data", config.dataset + ".txt")), ...
    sprintf("\r\n\r\n"), "   ");
% extract inputs and outputs
[words, idx] = unique(split(text(1:5:end - 5)));
inputs = cellfun(@double, words, "Un", 0);
outputs = cellfun(@(x) categorical(x - '0'), split(text(3:5:end - 5)), "Un", 0);
outputs = outputs(idx, :);
% generate train/test indices
[train, ~, test] = dividerand(numel(words), 0.7, 0, 0.3);
%% model
layers = [
    sequenceInputLayer(1)
    wordEmbeddingLayer(config.layers.embedding.dimension, numel(train))
    dropoutLayer(config.layers.dropout.probability)
    bilstmLayer(config.layers.bilstm.units)
    repmat([
        convolution1dLayer(config.layers.cnn.size, ...
            config.layers.cnn.filters, 'Padding', 'same')
        reluLayer
        maxPooling1dLayer(config.layers.cnn.size, 'Padding', 'same')], ...
    [config.layers.cnn.layers 1])
    concatenationLayer(1, 2)
    fullyConnectedLayer(19)
    softmaxLayer
    classificationLayer];
graph = layerGraph(layers);
graph = disconnectLayers(graph, "biLSTM", "conv1d_1");
graph = connectLayers(graph, "word-embedding", "conv1d_1");
graph = connectLayers(graph, "biLSTM", "concat/in2");
analyzeNetwork(graph);
%%
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'auto', ...
    'GradientThreshold', 1, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 64, ...
    'SequenceLength', 'longest', ...
    'Shuffle', 'never', ...
    'Verbose', true, ...
    'Plots', 'training-progress');
net = trainNetwork(inputs(train), outputs(train), graph, options);
