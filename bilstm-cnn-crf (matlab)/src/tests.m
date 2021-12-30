%% configuration
clear;
config = jsondecode(fileread("config.json"));
%% data
% load dataset
text = fileread(fullfile("data", config.datasets.names{1} + ".txt"));
% extract and preprocess inputs and outputs
breaks = strfind(text, (sprintf("\r\n\r\n")));
lengths = diff([-5 breaks + (1:3:numel(breaks) * 3)] / 5) - 1;
text([breaks breaks + 1]) = [];
inputs = mat2cell(double(text(1:5:end)), 1, lengths);
outputs = mat2cell(categorical(text(3:5:end) - '0'), 1, lengths);
% remove duplicate words
[~, keep] = unique(mat2cell(text(1:5:end), 1, lengths));
[inputs, outputs] = deal(inputs(keep), outputs(keep));
% generate train/validation/test indices
[training, validation, testing] = dividerand(numel(inputs), ...
    config.datasets.split.training, config.datasets.split.validate, ...
    config.datasets.split.testing);
%% model
layers = [
    sequenceInputLayer(1)
    wordEmbeddingLayer(config.layers.embedding.dimension, numel(training))
    dropoutLayer(config.layers.dropout.probability)
    bilstmLayer(config.layers.bilstm.units)
    repmat([
        convolution1dLayer(config.layers.cnn.size, ...
            config.layers.cnn.filters, "Padding", "same")
        reluLayer
        maxPooling1dLayer(config.layers.cnn.size, "Padding", "same")], ...
    [config.layers.cnn.layers 1])
    concatenationLayer(1, 2)
    fullyConnectedLayer(19)
    softmaxLayer
    classificationLayer];
graph = layerGraph(layers);
graph = disconnectLayers(graph, "biLSTM", "conv1d_1");
graph = connectLayers(graph, "word-embedding", "conv1d_1");
graph = connectLayers(graph, "biLSTM", "concat/in2");
% analyzeNetwork(graph);
%%
options = trainingOptions("adam", ...
    "ExecutionEnvironment", "auto", ...
    "GradientThreshold", 1, ...
    "MaxEpochs", config.training.maxepochs, ...
    "ValidationData", {inputs(validation), outputs(validation)}, ...
    "ValidationFrequency", 50, ...
    "ValidationPatience", 6, ...
    "MiniBatchSize", config.training.minibatchsize, ...
    "SequenceLength", "longest", ...
    "Shuffle", "never", ...
    "Verbose", false, ...
    "Plots", "training-progress");
net = trainNetwork(inputs(training), outputs(training), graph, options);
