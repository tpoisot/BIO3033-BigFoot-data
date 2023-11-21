function backwardselection(model, y, X, folds, perf, args...; kwargs...)
    pool = collect(axes(X, 1))
    best_perf = -Inf
    while ~isempty(pool)
        @info "N = $(length(pool))"
        scores = zeros(length(pool))
        Threads.@threads for i in eachindex(pool)
            this_pool = deleteat!(copy(pool), i)
            scores[i] = mean(perf.(first(crossvalidate(model, y, X[this_pool,:], folds, args...; kwargs...))))
        end
        best, i = findmax(scores)
        if best > best_perf
            best_perf = best
            deleteat!(pool, i)
        else
            @info "Returning with $(pool) -- $(best_perf)"
            break
        end
    end
    return pool
end

function constrainedselection(model, y, X, folds, pool, perf, args...; kwargs...)
    on_top = filter(p -> !(p in pool), collect(axes(X, 1)))
    best_perf = -Inf
    while ~isempty(on_top)
        @info "N = $(length(pool)+1)"
        scores = zeros(length(on_top))
        for i in eachindex(on_top)
            this_pool = push!(copy(pool), on_top[i])
            scores[i] = mean(perf.(first(crossvalidate(model, y, X[this_pool,:], folds, args...; kwargs...))))
        end
        best, i = findmax(scores)
        if best > best_perf
            best_perf = best
            push!(pool, on_top[i])
            deleteat!(on_top, i)
        else
            @info "Returning with $(pool) -- $(best_perf)"
            break
        end
    end
    return pool
end

function forwardselection(model, y, X, folds, perf, args...; kwargs...)
    pool = Int64[]
    constrainedselection(model, y, X, folds, pool, perf, args...; kwargs...)
end