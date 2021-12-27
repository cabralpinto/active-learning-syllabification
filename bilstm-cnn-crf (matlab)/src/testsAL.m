%% configuration
config = jsondecode(fileread("configAL.json"));
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
            config.layers.cnn.filters, 'Padding', 'same')
        reluLayer
        maxPooling1dLayer(2, 'Padding', 'same')], ...
    [config.layers.cnn.layers 1])
    concatenationLayer(1, 2)
    fullyConnectedLayer(size(outputs, 2)) % time distributed?
    softmaxLayer
    ];
graph = layerGraph(layers);
graph = disconnectLayers(graph, "biLSTM", "conv1d_1");
graph = connectLayers(graph, "word-embedding", "conv1d_1");
graph = connectLayers(graph, "biLSTM", "concat/in2");

dlnet = dlnetwork(graph);

%% training
X_train = inputs(train);
Y_train = outputs(train, :);

velocity = [];
iteration = 0;

figure
lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
ylim([0 inf])
xlabel("Iteration")
ylabel("Loss")
grid on

for epoch = 1:config.training.epochs
    for i = 1:numel(X_train)
        iteration = iteration + 1;

        x = X_train(i:i+5);
        y = Y_train(:, i:i+5);

        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,x,y);
        dlnet.State = state;

        learnRate = config.training.initialLearnRate / ...
            (1 + config.training.decay*iteration);
        
        [dlnet,velocity] = sgdmupdate(dlnet,gradients,velocity,...
            config.training.learnRate, ...
            config.training.momentum);

        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(lineLossTrain,iteration,loss)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end
