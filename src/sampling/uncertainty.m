function uncertainty = uncertainty(prediction)
uncertainty = mean(1 - max(prediction, [], 1));
end

