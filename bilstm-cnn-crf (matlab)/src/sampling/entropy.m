function entropy = entropy(prediction)
entropy = -mean(sum(prediction .* log(prediction + (prediction == 0)), 1));
end

