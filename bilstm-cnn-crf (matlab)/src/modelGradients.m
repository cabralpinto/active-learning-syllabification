function [gradients,state,loss] = modelGradients(dlnet,dlX,Y)

[dlYPred,state] = forward(dlnet,dlX);

loss = crossentropy(dlYPred,Y);
gradients = dlgradient(loss,dlnet.Learnables);

loss = double(gather(extractdata(loss)));

end