using Comrade 
using Pyehtim
using Serialization

function convert_to_com(file, out, dataproduct)
    obs = ehtim.obsdata.load_uvfits(file)
    vis = extract_table(obs, dataproduct)
    serialize(out, vis)
end

#files = filter(endswith(".uvfits"), readdir("noisefrac0.02grmhd3599", join=true))
files = filter(endswith(".uvfits"), readdir("uvfits/hops_3598_MAD_a-0.5_Rh40_i50_LO/snapshot_120/noisefrac0.02/hops_3598_MAD_a-0.5_Rh40_i50_LO/", join=true))


mkpath("data_grmhdtest3598")

outs = joinpath.("data_grmhdtest3598", replace.(basename.(files), ".uvfits" => ".jls"))

convert_to_com.(files, outs, Ref(Visibilities()))

println(outs)
println(outs[3])
# Now if Comrade is loaded we can load the data with (you do not need Pyehtim loaded for this)
deserialize(outs[3])