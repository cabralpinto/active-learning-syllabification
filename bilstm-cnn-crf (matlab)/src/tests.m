% clear workspace
clear;
% add sampling functions to path
addpath("src/sampling");
% load config
config = jsondecode(fileread("config.json"));
% create results table
results = table(rownames = config.datasets.names);
% testing loop
for dataset = config.datasets.names'
    % load dataset
    text = fileread(fullfile("data", dataset{:} + ".txt"));
    % extract and preprocess inputs and outputs
    breaks = strfind(text, (sprintf("\r\n\r\n")));
    lengths = diff([-5 breaks + (1:3:numel(breaks) * 3)] / 5) - 1;
    text([breaks breaks + 1]) = [];
    inputs = mat2cell(double(text(1:5:end)), 1, lengths);
    outputs = mat2cell(categorical(text(3:5:end) - '0'), 1, lengths);
    % remove duplicate words
    [~, keep] = unique(mat2cell(text(1:5:end), 1, lengths));
    [inputs, outputs, lengths] = deal(inputs(keep), outputs(keep), ...
        lengths(keep));
    % sort words by length
    [~, sorted] = sort(lengths);
    [inputs, outputs] = deal(inputs(sorted), outputs(sorted));
    % generate train/validation/test indices
    [training, validation, testing] = dividerand(numel(inputs), ...
        config.datasets.split.training, config.datasets.split.validate, ...
        config.datasets.split.testing);
    % create model
    layers = [
        sequenceInputLayer(1)
        wordEmbeddingLayer(config.layers.embedding.dimension, ...
            numel(training))
        dropoutLayer(config.layers.dropout.probability)
        bilstmLayer(config.layers.bilstm.units)
        repmat([
            convolution1dLayer(config.layers.cnn.size, ...
                config.layers.cnn.filters, padding = "same")
            reluLayer
            maxPooling1dLayer(config.layers.cnn.size, ...
                padding = "same")], ...
            [config.layers.cnn.repeat 1])
        concatenationLayer(1, 2)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer];
    model = layerGraph(layers);
    model = disconnectLayers(model, "biLSTM", "conv1d_1");
    model = connectLayers(model, "word-embedding", "conv1d_1");
    model = connectLayers(model, "biLSTM", "concat/in2");
    % set training options
    options = trainingOptions("adam", ... outputfcn = @(info) true, ... % early stop
        maxepochs = config.training.maxepochs, ...
        validationdata = {inputs(validation), outputs(validation)}, ...
        validationfrequency = config.training.validationfrequency, ...
        validationpatience = config.training.validationpatience, ...
        outputnetwork = "best-validation-loss", ...
        minibatchsize = config.training.minibatchsize, ...
        sequencelength = "longest", ...
        shuffle = "never", ...
        plots = "training-progress", ...
        verbose = false, ...
        checkpointpath = "nets");
    % train model on full dataset and test
    net = trainNetwork(inputs(training), outputs(training), model, ...
        options);
    classifications = classify(net, inputs(testing), ...
        minibatchsize = config.training.minibatchsize)';
    accuracy = sum(cellfun(@isequal, classifications, ...
        outputs(testing))) / numel(testing);
    fprintf("Full dataset training accuracy: %.3g%%\n", accuracy * 100);
    % active learning testing loop
    for strategy = {@uncertainty @margin @entropy}
        % determine initial training dataset
        initial = randperm(numel(training), fix(numel(training) * ...
            config.training.initialpercentage));
        % train model on initial training dataset
        net = trainNetwork(inputs(training(initial)), ...
            outputs(training(initial)), model, options);
        % remove initial dataset from remaining training indices
        remaining = 1:numel(training);
        remaining(initial) = [];
        % active learning loop
        for query = 0:config.training.queries
            % test network
            classifications = classify(net, inputs(testing), ...
                minibatchsize = config.training.minibatchsize)';
            accuracy = sum(cellfun(@isequal, classifications, ...
                outputs(testing))) / numel(testing);
            fprintf("Active learning accuracy after %d queries: " + ...
                "%.3g%%\n", query, accuracy * 100);
            % end on last query
            if query == config.training.queries, break; end
            % get network predictions on remaining training samples
            predictions = predict(net, inputs(training(remaining)), ...
                minibatchsize = config.training.minibatchsize)';
            % query remaining training samples
            [~, queried] = maxk(cellfun(strategy{:}, predictions), ...
                config.training.minibatchsize);
            % train network on queried samples
            net = trainNetwork(inputs(training(remaining(queried))), ...
                outputs(training(remaining(queried))), layerGraph(net), ...
                options);
            % remove queried instances from remaining training set
            remaining(queried) = [];
        end
    end
end

