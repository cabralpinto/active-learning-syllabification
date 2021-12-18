%% configuration
config = jsondecode(fileread("config.json"));
%% data
% load dataset
text = insertAfter(fileread(fullfile("data", config.dataset + ".txt")), ...
    sprintf("\r\n\r\n"), "   ");
% extract inputs and outputs
inputs = split(string(text(1:5:end - 5)));
outputs = cell2mat(pad(split(text(3:5:end - 5)), '0')) - '0';
% generate train/test indices
[train, ~, test] = dividerand(numel(inputs), 0.7, 0, 0.3);
%% model
layers = [
    sequenceInputLayer(1)
    wordEmbeddingLayer(config.layers.embedding.dimension, ...
        numel(inputs(train)))
    dropoutLayer(config.layers.dropout.probability)
    bilstmLayer(config.layers.bilstm.units)
    repmat([
        convolution1dLayer(config.layers.cnn.size, ...
            config.layers.cnn.filters)
        reluLayer
        globalMaxPooling1dLayer], ...
    [config.layers.cnn.layers 1])
    concatenationLayer(1, 2)
    fullyConnectedLayer(2 * size(outputs, 2)) % time distributed?
    softmaxLayer
    classificationLayer];
graph = layerGraph(layers);
graph = disconnectLayers(graph, "biLSTM", "conv1d_1");
graph = connectLayers(graph, "word-embedding", "conv1d_1");
graph = connectLayers(graph, "biLSTM", "concat/in2");
analyzeNetwork(graph);