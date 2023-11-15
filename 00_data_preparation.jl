import DelimitedFiles
using SpeciesDistributionToolkit

sightings = vec(mapslices(x -> Tuple(reverse(x)), DelimitedFiles.readdlm("occurrences.csv", ','); dims=2))
filter!(r -> 25 <= r[2] <= 55, sightings)
filter!(r -> -130 <= r[1] <= -70, sightings)

boundingbox = (
    left=-130.,
    right=-70.,
    bottom=25.,
    top=55.,
)

provider = RasterData(WorldClim2, BioClim)
opts = (; resolution=2.5)
temperature = SimpleSDMPredictor(provider, layer=1; opts..., boundingbox...)

water = SimpleSDMPredictor(RasterData(EarthEnv, LandCover), layer=12; boundingbox...)
land = similar(temperature, Bool)
replace!(land, false => true)
Threads.@threads for k in keys(land)
    if !isnothing(water[k])
        if water[k] == 100
            land[k] = false
        end
    end
end
temperature = mask(land, temperature)

presence_layer = similar(temperature, Bool)
for i in axes(sightings, 1)
    if ~isnothing(presence_layer[sightings[i]...])
        presence_layer[sightings[i]...] = true
    end
end

possible_background =
    pseudoabsencemask(DistanceToEvent, presence_layer) *
    cellsize(temperature)

absence_layer = backgroundpoints(
    (x -> x^1.01).(possible_background), 
    2sum(presence_layer);
    replace=false
)

replace!(absence_layer, false => nothing)
replace!(presence_layer, false => nothing)

predictors = 1:19
Xpresence = hcat([bioclim_var[keys(presence_layer)] for bioclim_var in predictors]...)
ypresence = fill(true, length(presence_layer))
Xabsence = hcat([bioclim_var[keys(absence_layer)] for bioclim_var in predictors]...)
yabsence = fill(false, length(absence_layer))
X = vcat(Xpresence, Xabsence)
y = vcat(ypresence, yabsence)

idx, tidx = holdout(y, X; permute=true)