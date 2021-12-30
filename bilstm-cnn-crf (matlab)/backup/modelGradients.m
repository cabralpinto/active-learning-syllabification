function [gradients,state,loss] = modelGradients(dlnet,x,y)

[yPred,state] = forward(dlnet,x);
yPred = yPred(:, 1); % LOL CHANGE THIS

loss = crossentropy(yPred,y');
gradients = dlgradient(loss,dlnet.Learnables);

loss = double(gather(extractdata(loss)));

end