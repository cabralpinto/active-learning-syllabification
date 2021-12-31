function margin = margin(prediction)
margin = mean(diff(maxk(prediction, 2, 1)));
end

