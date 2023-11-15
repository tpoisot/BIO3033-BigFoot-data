function noskill(labels::Vector{Bool})
    n = length(labels)
    p = mean(labels)
    tp = p^2
    tn = (1 - p)^2
    fp = p * (1 - p)
    fn = (1 - p) * p
    return ConfusionMatrix(tp, tn, fp, fn)
end

function coinflip(labels::Vector{Bool})
    n = length(labels)
    p = mean(labels)
    tp = 1 / 2 * p
    tn = 1 / 2 * p
    fp = 1 / 2 * (1 - p)
    fn = 1 / 2 * (1 - p)
    return ConfusionMatrix(tp, tn, fp, fn)
end