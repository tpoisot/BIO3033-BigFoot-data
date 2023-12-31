# Types to fit
types_to_train = [:PCA, :PPCA, :KernelPCA, :Whitening, :MDS, :MetricMDS]
types_with_transform = [:Whitening]

Base.@kwdef mutable struct MultivariateTransform{T} <: SDMTransformer
    trf::T = StatsAPI.fit(T, rand(0:10, (8, 8)))
end

for tf in types_to_train
    fitpkg = tf in types_with_transform ? :MultivariateStats : :StatsAPI
    fitfunc = tf in types_with_transform ? :transform : :predict
    eval(
        quote
            function train!(trf::MultivariateTransform{$tf}, X; kwdef...)
                trf.trf = StatsAPI.fit(MultivariateStats.$tf, X)
            end
            function StatsAPI.predict(trf::MultivariateTransform{$tf}, x::AbstractArray; kwdef...)
                return $(fitpkg).$(fitfunc)(trf.trf, x)
            end
            
        end
    )
end
