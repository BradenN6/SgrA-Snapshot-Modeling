using EHTModelStacker
using DataFrames
using CSV


function load_chains(dir, outdir)
    skychain_paths = filter(endswith("skychain.csv"), readdir(dir, join=true))
    println(skychain_paths)
    println(length(skychain_paths))
    sum_paths = filter(endswith("dfsum.csv"), readdir(dir, join=true))
    println(length(sum_paths))

    #dfchain = vcat([CSV.read(f, DataFrame) for f in skychain_paths]...)
    dfchain = [CSV.read(f, DataFrame) for f in skychain_paths]
    dfsum = vcat([CSV.read(f, DataFrame) for f in sum_paths]...)

    #println(dfchain[1])
    println(length(dfchain))

    CSV.write(joinpath(outdir, "dfchain.csv"), dfchain)
    CSV.write(joinpath(outdir, "dfsum.csv"), dfsum)

    return dfchain, dfsum
end


function make_hdf5_chain(dir, outdir, outname)
    dfchain, dfsum = load_chains(dir, outdir)
    EHTModelStacker.write2h5(dfchain, dfsum, outname)
end

#dir = joinpath((@__DIR__), "./../csv-chains-grmhdtest3599-rad/")
#outdir = joinpath((@__DIR__), "df-csv-grmhdtest3599-rad")
dir = joinpath((@__DIR__), "./../csv-chains-grmhdtest3599/")
outdir = joinpath((@__DIR__), "df-csv-grmhdtest3599")
mkpath(outdir)
outname = joinpath(outdir, "stacker_chain_grmhdtest3599.h5")

make_hdf5_chain(dir, outdir, outname)




# TODO update to use actual scan numbers
#=
function write2h5(dfchain, dfsum, outname)
    h5open(outname, "w") do fid
        fid["time"] = dfsum[:,:time]
        fid["logz"] = dfsum[:,:logz]
        pid = create_group(fid, "params")
        for i in 1:length(dfchain)
            sid = create_group(pid, "scan$i")
            keys = names(dfchain[i])
            for k in keys
                write(sid, k, dfchain[i][:,Symbol(k)])
            end
        end
    end
end
=#

# TODO handle mixturemodel in priorKrang.txt?
# TODO angular variables - pi exact?
# TODO angular for incl, opening angle
# TODO restrict std