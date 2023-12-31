using MLJ
import JLD2
using DataFrames
using SpeciesDistributionToolkit
using GeoMakie
using CairoMakie

# Load the data
JLD2.jldopen("training.jld2", "r") do file
    global X = file["X"]
    global y = file["y"]
end;
presences = findall(y);
absences = findall(.!y);

# Prepare the data as a large dataframe
df = DataFrame(X', "BIO".*string.(1:19))
df.presence = y
df.presence = coerce(df.presence, OrderedFactor)

# Check the schema
schema(df)

# Prepare for MLJ
y, X = unpack(df, ==(:presence); rng=123);

# List the compatible models
for model in models(matching(X, y))
    @info model
end

# Load a few tree-based classifiers
#Classifier = MLJ.@load DecisionTreeClassifier pkg=DecisionTree
Classifier = MLJ.@load RandomForestClassifier pkg=DecisionTree
#Classifier = MLJ.@load EvoTreeClassifier pkg=EvoTrees

# Bind the data to the machine
classifier = Classifier()
M = machine(classifier, X, y)

# Evaluate
evaluate!(M, resampling=CV(nfolds=10, shuffle=true, rng=1234), measure=[MatthewsCorrelation(), NegativePredictiveValue(), PositivePredictiveValue()])

fit!(M)

# Load the data
layers = [SpeciesDistributionToolkit._read_geotiff("layers.tiff", SimpleSDMPredictor; bandnumber=v) for v in 1:19]
prediction = similar(layers[1])
D = zeros(Float32, (length(layers), length(layers[1])))
Threads.@threads for i in eachindex(layers)
    D[i,:] = values(layers[i])
end

# Predict
df2 = DataFrame(D', "BIO".*string.(1:19))
yhat = predict(M, df2)
prediction.grid[findall(!isnothing, prediction.grid)] = pdf.(yhat, true)

# Load the occurrence data
sightings = map(x -> tuple(reverse(parse.(Float64, x))...), split.(readlines("occurrences.csv"), ','))
filter!(r -> 25 <= r[2] <= 55, sightings);
filter!(r -> -130 <= r[1] <= -70, sightings);

# Plot
fig = Figure()
ax = GeoMakie.GeoAxis(fig[1,1]; dest = "+proj=aea +lat_1=0.0 +lat_2=55.0", coastlines=true, lonlims=extrema(longitudes(prediction)), latlims=extrema(latitudes(prediction)))
surface!(ax, prediction; colormap=:lapaz, colorrange=(0,1), shading=false)
scatter!(ax, sightings, color=:orange, markersize=3)
hidedecorations!(ax)
hidespines!(ax)
current_figure()

#=
# yeah booooooistrap
using StatsBase
bags = [sample(train, length(train); replace=true) for i in 1:40]
Y = zeros(Float64, (length(bags), size(df2, 1)))
Threads.@threads for i in eachindex(bags)
    fit!(M, rows=bags[i])
    Y[i,:] .= pdf.(predict(M, df2), true)
end

bsmedian = similar(prediction)
bsmean = similar(prediction)
bsvariance = similar(prediction)
bsmean.grid[findall(!isnothing, prediction.grid)] = vec(mapslices(mean, Y; dims=1))
bsmedian.grid[findall(!isnothing, prediction.grid)] = vec(mapslices(median, Y; dims=1))
bsvariance.grid[findall(!isnothing, prediction.grid)] = vec(mapslices(var, Y; dims=1))

bsz = (prediction - bsmean)/bsvariance

fig = Figure()
ax = GeoMakie.GeoAxis(fig[1,1]; dest = "+proj=aea +lat_1=0.0 +lat_2=55.0", coastlines=true, lonlims=extrema(longitudes(prediction)), latlims=extrema(latitudes(prediction)))
surface!(ax, bsvariance; colormap=:lapaz, colorrange=(0,1), shading=false)
scatter!(ax, sightings, color=:orange, markersize=3)
hidedecorations!(ax)
hidespines!(ax)
current_figure()
=#

# Future
layers_future = [SpeciesDistributionToolkit._read_geotiff("layers-2090.tiff", SimpleSDMPredictor; bandnumber=v) for v in 1:19]
future_prediction = similar(layers_future[1])
D = zeros(Float32, (length(layers_future), length(layers_future[1])))
Threads.@threads for i in eachindex(layers_future)
    D[i,:] = values(layers_future[i])
end

# Predict
df2 = DataFrame(D', "BIO".*string.(1:19))
yhat = predict(M, df2)
future_prediction.grid[findall(!isnothing, future_prediction.grid)] = pdf.(yhat, true)

fig = Figure()
ax = GeoMakie.GeoAxis(fig[1,1]; dest = "+proj=aea +lat_1=0.0 +lat_2=55.0", coastlines=true, lonlims=extrema(longitudes(prediction)), latlims=extrema(latitudes(prediction)))
surface!(ax, future_prediction; colormap=:lapaz, colorrange=(0,1), shading=false)
scatter!(ax, sightings, color=:orange, markersize=3)
hidedecorations!(ax)
hidespines!(ax)
current_figure()

# 
prediction.left = future_prediction.left
prediction.right = future_prediction.right
prediction.bottom = future_prediction.bottom
prediction.top = future_prediction.top


heatmap(future_prediction - prediction, colormap=:Spectral, colorrange=(-0.5, 0.5))