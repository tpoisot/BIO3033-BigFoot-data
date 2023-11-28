function Base.show(io::IO, ensemble::Bagging)
    strs = [
        "An ensemble of $(length(ensemble.models)) models",
        "\tbase model: $(ensemble.model)"
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