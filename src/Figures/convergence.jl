using JLD2, Pigeons, DataFrames, CairoMakie
using Glob

data_dir = "/Users/bnowicki/Research/CfA2025-Paper-Chains/pt-objects"

# Find every *-pt.jld2 file in the folder
pt_files = glob("*-pt.jld2", data_dir)

# Extract the scan ID from each filename (e.g. "scan136-pt.jld2" -> "scan136")
scan_ids = [replace(basename(f), "-pt.jld2" => "") for f in pt_files]

println("Found $(length(scan_ids)) scans:")
foreach(println, scan_ids)

results = DataFrame(scan=String[], round=Int[], logZ=Float64[])

for scan in scan_ids
    try
        pt = JLD2.load(joinpath(data_dir, "$scan-pt.jld2"), "pt")
        summary = pt.shared.reports.summary
        for row in eachrow(summary)
            push!(results, (scan, row.round, row.stepping_stone))
        end
    catch e
        @warn "Failed to load $scan" exception=e
    end
end

# logZ0 = the final round's evidence estimate for each chain
logZ0 = combine(groupby(results, :scan), :logZ => last => :logZ0)
results = leftjoin(results, logZ0, on=:scan)
results.delta = results.logZ .- results.logZ0

fig = Figure()
#ax = Axis(fig[1,1], xlabel="Round", ylabel="log Z − log Z₀")

ax = Axis(fig[1,1], xlabel="Round", ylabel="log Z − log Z₀",
          xticks = 1:maximum(results.round),
          xtickformat = xs -> string.(round.(Int, xs)))

for scan in unique(results.scan)
    sub = sort(results[results.scan .== scan, :], :round)
    lines!(ax, sub.round, sub.delta, label=scan)
    scatter!(ax, sub.round, sub.delta)
end


hlines!(ax, [0.0], color=:black, linestyle=:dash)
#axislegend(ax, position=:rt)
fig