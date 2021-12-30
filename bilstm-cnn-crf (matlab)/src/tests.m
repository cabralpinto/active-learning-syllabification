clear;
%% configuration
config = jsondecode(fileread("config.json"));
%% dataset
% load dataset
text = fileread(fullfile("data", config.datasets.names{1} + ".txt"));
% extract and preprocess inputs and outputs
breaks = strfind(text, (sprintf("\r\n\r\n")));
lengths = diff([-5 breaks + (1:3:numel(breaks) * 3)] / 5) - 1;
text([breaks breaks + 1]) = [];
inputs = mat2cell(double(text(1:5:end)), 1, lengths);
outputs = mat2cell(categorical(text(3:5:end) - '0'), 1, lengths);
% remove duplicate words
[~, idx] = unique(mat2cell(text(1:5:end), 1, lengths));
[inputs, outputs, lengths] = deal(inputs(idx), outputs(idx), lengths(idx));
% sort words by length
[lengths, idx] = sort(lengths);
[inputs, outputs] = deal(inputs(idx), outputs(idx));
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
    fullyConnectedLayer(2 * lengths(end))
    softmaxLayer
    classificationLayer];
model = layerGraph(layers);
model = disconnectLayers(model, "biLSTM", "conv1d_1");
model = connectLayers(model, "word-embedding", "conv1d_1");
model = connectLayers(model, "biLSTM", "concat/in2");
% analyzeNetwork(graph);
%% training
options = trainingOptions("adam", ...
    "MaxEpochs", config.training.maxepochs, ...
    "ValidationData", {inputs(validation), outputs(validation)}, ...
    "ValidationFrequency", config.training.validationfrequency, ...
    "ValidationPatience", config.training.validationpatience, ...
    "OutputNetwork", "best-validation-loss", ...
    "MiniBatchSize", config.training.minibatchsize, ...
    "SequenceLength", "longest", ...
    "Shuffle", "never", ...
    "Verbose", true);
net = trainNetwork(inputs(training), outputs(training), model, options);
%% testing
predictions = classify(net, inputs(testing), ...
    "MiniBatchSize", config.training.minibatchsize);
accuracy = sum(cellfun(@isequal, predictions, outputs(testing)')) / ...
    numel(testing);