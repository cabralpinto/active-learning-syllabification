% settings
dataset = "french";
% load dataset
text = insertAfter(fileread(fullfile("data", dataset + ".txt")), ...
    sprintf("\r\n\r\n"), "   ");
words = split(string(text(1:5:end - 5)));
targets = cell2mat(pad(split(text(3:5:end - 5)), '0')) - '0';