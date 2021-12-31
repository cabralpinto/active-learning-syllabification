% clear workspace
clear;
% add sampling functions to path
addpath("src/sampling");
% load config
config = jsondecode(fileread("config.json"));
% testing loop
for dataset = config.datasets.names'
    % print datset header
    fprintf("> %s\n", dataset{:});
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
    options = trainingOptions("adam", ... OutputFcn = @(info) true, ... % early stop
        MaxEpochs = config.training.maxepochs, ...
        ValidationData = {inputs(validation), outputs(validation)}, ...
        ValidationFrequency = config.training.validationfrequency, ...
        ValidationPatience = config.training.validationpatience, ...
        OutputNetwork = "best-validation-loss", ...
        MiniBatchSize = config.training.minibatchsize, ...
        SequenceLength = "longest", ...
        Shuffle = "never", ...
        Verbose = false);
    % print active learning header
    fprintf("  > traditional learning (word count: %d)\n", ...
        numel(training));
    % train model on full dataset and test
    net = trainNetwork(inputs(training), outputs(training), model, ...
        options);
    % test model
    classifications = classify(net, inputs(testing), ...
        minibatchsize = config.training.minibatchsize)';
    accuracy = sum(cellfun(@isequal, classifications, ...
        outputs(testing))) / numel(testing);
    fprintf("    > accuracy: %.3g%%\n", accuracy * 100);
    % only validate at the beggining and end
    options.ValidationFrequency = 99999;
    % print active learning header
    fprintf("  > active learning (initial word count: %d)\n", ...
        config.training.initialsize);
    % active learning testing loop
    for strategy = {@entropy @margin @uncertainty @rand}
        % print query strategy header
        fprintf("    > %s sampling\n", func2str(strategy{:}));
        % initialize selected training indices
        selected = false(numel(training), 1);
        selected(randperm(numel(training), ...
            config.training.initialsize)) = true;
        % train model on random words
        net = trainNetwork(inputs(training(selected)), ...
            outputs(training(selected)), model, options);
        % initialize best accuracy
        bestaccuracy = 0;
        % active learning loop
        for query = 0:inf
            % test model
            classifications = classify(net, inputs(testing), ...
                minibatchsize = config.training.minibatchsize)';
            accuracy = sum(cellfun(@isequal, classifications, ...
                outputs(testing))) / numel(testing);
            fprintf("      > %d queries: %.3g%%\n", query, accuracy * 100);
            % update best accuracy
            bestaccuracy = max([bestaccuracy accuracy]);
            % break condition
            if query == config.training.maxqueries || ...
                query >= config.training.minqueries && ...
                bestaccuracy > config.training.accuracythreshold
                break;
            end
            % get model predictions on remaining training samples
            predictions = predict(net, inputs(training(~selected)), ...
                minibatchsize = config.training.minibatchsize)';
            % query remaining training samples
            [~, queried] = maxk(cellfun(strategy{:}, predictions), ...
                1);
            trainingnotselected = training(~selected);
            selected(trainingnotselected(queried)) = true;
            % retrain existing model on queried samples
            net = trainNetwork(inputs(training(selected)), ...
                outputs(training(selected)), layerGraph(net), options);
        end
    end
end

