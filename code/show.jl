function Base.show(io::IO, ensemble::Bagging)
    strs = [
        "{$(ensemble.model)} × $(length(ensemble.models))"
    ]
    print(io, join(strs, "\n"))
end

function Base.show(io::IO, sdm::SDM)
    strs = [
        "$(typeof(sdm.transformer))",
        "$(typeof(sdm.classifier))",
        "P(x) ≥ $(round(sdm.threshold.cutoff; digits=3))",
    ]
    print(io, join(strs, " → "))
end

function Base.show(io::IO, C::ConfusionMatrix)
    str = "[TP: $(C.tp), TN $(C.tn), FP $(C.fp), FN $(C.fn)]"
    print(io, str)
end
