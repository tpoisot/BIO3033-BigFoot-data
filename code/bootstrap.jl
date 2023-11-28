function bootstrap(y, X; n=50)
    @assert size(y, 1) == size(X, 2)
    bags = []
    for _ in 1:n
        inbag = sample(1:size(X, 2), size(X, 2); replace=true)
        outbag = setdiff(axes(X, 2), inbag)
        push!(bags, (inbag, outbag))
    end
    return bags
end

mutable struct Bagging
    model::SDM
    bags::Vector{Tuple{Vector{Int64},Vector{Int64}}}
    models::Vector{SDM}
end

function Bagging(model::SDM, bags::Vector)
    return Bagging(model, bags, typeof(model)[])
end

function train!(ensemble::Bagging, y, X; kwargs...)
    ensemble.models = [train!(deepcopy(ensemble.model), y, X[:, bag[1]]; kwargs...) for bag in ensemble.bags]
    return ensemble
end

function StatsAPI.predict(ensemble::Bagging, X; consensus=median, kwargs...)
    ŷ = [predict(component, X; kwargs...) for component in ensemble.models]
    ỹ = vec(mapslices(consensus, hcat(ŷ...); dims=2))
    return isone(length(ỹ)) ? only(ỹ) : ỹ
end

function outofbag(ensemble::Bagging, y, X, bags, args...; kwargs...)
    instance = rand(eachindex(y))
    done_instances = Int64[]
    outcomes = Bool[]

    for instance in eachindex(y)
        valid_models = findall(x -> !(instance in x[1]), bags)
        if !isempty(valid_models)
            push!(done_instances, instance)
            pred = [predict(ensemble.models[i], X[variables, instance], args...; kwargs...) for i in valid_models]
            push!(outcomes, count(pred) > count(pred) // 2)
        end
    end

    return ConfusionMatrix(outcomes, y[done_instances])
end